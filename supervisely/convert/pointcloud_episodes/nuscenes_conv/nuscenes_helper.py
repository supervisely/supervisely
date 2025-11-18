from dataclasses import dataclass, field
from datetime import datetime
from os import path as osp
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np

from supervisely import fs, logger
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d

DIR_NAMES = [
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "LIDAR_TOP",
    # "RADAR_FRONT",
    # "RADAR_FRONT_LEFT",
    # "RADAR_FRONT_RIGHT",
    # "RADAR_BACK_LEFT",
    # "RADAR_BACK_RIGHT",
]

TABLE_NAMES = [
    "category",
    "attribute",
    "visibility",
    "instance",
    "sensor",
    "calibrated_sensor",
    "ego_pose",
    "log",
    "scene",
    "sample",
    "sample_data",
    "sample_annotation",
    "map",
]


def trim_description(description: str, max_length: int = 255) -> str:
    if len(description) > max_length:
        sentences = description.split(".")
        trimmed_description = ""
        for sentence in sentences:
            if len(trimmed_description) + len(sentence) + 1 > max_length:
                break
            trimmed_description += sentence + "."
        description = trimmed_description.strip()
    return description


@dataclass
class AnnotationObject:
    """
    A class to represent an annotation object in the NuScenes dataset.

    :param name: The name of the annotation object
    :type name: str
    :param bbox: The bounding box coordinates in NuScenes format
    :type bbox: np.ndarray
    :param token: The unique token identifying the annotation object
    :type token: str
    :param instance_token: The instance token associated with the annotation object
    :type instance_token: str
    :param parent_token: The token of instance preceding the current object instance
    :type parent_token: str
    :param category: The class name of the annotation object
    :type category: str
    :param attributes: The attribute names associated with the annotation object
    :type attributes: List[str]
    :param visibility: The visibility level of the annotation object
    :type visibility: str
    """
    name: str
    bbox: np.ndarray
    token: str
    instance_token: str
    parent_token: str
    category: str
    attributes: List[str]
    visibility: str

    def to_supervisely(self) -> Cuboid3d:
        box = self.convert_nuscenes_to_BEVBox3D()

        bbox = box.to_xyzwhlr()
        dim = bbox[[3, 5, 4]]
        pos = bbox[:3] + [0, 0, dim[1] / 2]
        yaw = bbox[-1]

        position = Vector3d(float(pos[0]), float(pos[1]), float(pos[2]))
        rotation = Vector3d(0, 0, float(-yaw))
        dimension = Vector3d(float(dim[0]), float(dim[2]), float(dim[1]))
        geometry = Cuboid3d(position, rotation, dimension)

        return geometry

    def convert_nuscenes_to_BEVBox3D(self):
        import open3d as o3d  # pylint: disable=import-error

        box = self.bbox
        center = [float(box[0]), float(box[1]), float(box[2])]
        size = [float(box[3]), float(box[5]), float(box[4])]
        ry = float(box[6])
        yaw = ry - np.pi
        yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi
        world_cam = None
        return o3d.ml.datasets.utils.BEVBox3D(center, size, yaw, self.name, -1.0, world_cam)


class CamData:
    """
    This class handles camera sensor data from the nuScenes dataset, including coordinate system
    transformations from lidar to camera space and extraction of camera calibration parameters.

    :param nuscenes: The nuScenes dataset instance
    :type nuscenes: NuScenes
    :param sensor_name: The name of the camera sensor
    :type sensor_name: str
    :param sensor_token: The token identifying the specific sensor sample
    :type sensor_token: str
    :param cs_record: The calibrated sensor record for the lidar
    :type cs_record: dict
    :param ego_record: The ego pose record for the lidar
    :type ego_record: dict
    """

    def __init__(
        self, nuscenes, sensor_name: str, sensor_token: str, cs_record: dict, ego_record: dict
    ):
        from nuscenes import NuScenes  # pylint: disable=import-error
        from nuscenes.utils.data_classes import (  # pylint: disable=import-error
            transform_matrix,
        )
        from pyquaternion import Quaternion  # pylint: disable=import-error

        nuscenes: NuScenes = nuscenes

        img_path, _, _ = nuscenes.get_sample_data(sensor_token)
        if not osp.exists(img_path):
            return None

        sd_record_cam = nuscenes.get("sample_data", sensor_token)
        cs_record_cam = nuscenes.get("calibrated_sensor", sd_record_cam["calibrated_sensor_token"])
        ego_record_cam = nuscenes.get("ego_pose", sd_record_cam["ego_pose_token"])
        lid_to_ego = transform_matrix(
            cs_record["translation"],
            Quaternion(cs_record["rotation"]),
            inverse=False,
        )
        lid_ego_to_world = transform_matrix(
            ego_record["translation"],
            Quaternion(ego_record["rotation"]),
            inverse=False,
        )
        world_to_cam_ego = transform_matrix(
            ego_record_cam["translation"],
            Quaternion(ego_record_cam["rotation"]),
            inverse=True,
        )
        ego_to_cam = transform_matrix(
            cs_record_cam["translation"],
            Quaternion(cs_record_cam["rotation"]),
            inverse=True,
        )
        velo_to_cam = np.dot(
            ego_to_cam, np.dot(world_to_cam_ego, np.dot(lid_ego_to_world, lid_to_ego))
        )
        velo_to_cam_rot = velo_to_cam[:3, :3]
        velo_to_cam_trans = velo_to_cam[:3, 3]

        self.name = sensor_name
        self.path = str(img_path)
        self.imsize = (sd_record_cam["width"], sd_record_cam["height"])
        self.extrinsic = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
        self.intrinsic = np.asarray(cs_record_cam["camera_intrinsic"])

    def get_info(self, timestamp: str) -> Tuple[str, Dict]:
        """
        Generates image info based on the camera data.

        :param timestamp: The timestamp associated with the image
        :type timestamp: str
        :return: A tuple containing the image path and a dictionary with image metadata.
        :rtype: tuple
        """
        sensors_to_skip = ["_intrinsic", "_extrinsic", "_imsize"]
        if not any([self.name.endswith(s) for s in sensors_to_skip]):
            image_name = fs.get_file_name_with_ext(self.path)
            sly_path_img = osp.join(osp.dirname(self.path), image_name)
            img_info = {
                "name": image_name,
                "meta": {
                    "deviceId": self.name,
                    "timestamp": timestamp,
                    "sensorsData": {
                        "extrinsicMatrix": list(self.extrinsic.flatten().astype(float)),
                        "intrinsicMatrix": list(self.intrinsic.flatten().astype(float)),
                    },
                },
            }
            return (sly_path_img, img_info)


@dataclass
class Sample:
    """
    A class to represent a sample from the NuScenes dataset.
    """
    timestamp_us: float
    lidar_path: str
    anns: List[AnnotationObject]
    cam_data: List[CamData]
    timestamp: str = field(init=False)

    def __post_init__(self):
        self.timestamp = datetime.utcfromtimestamp(self.timestamp_us / 1e6).isoformat()

    @staticmethod
    def generate_boxes(nuscenes, boxes: List) -> Generator:
        """
        Generate ground truth boxes for a given set of boxes.

        :param nuscenes: The nuScenes dataset instance
        :type nuscenes: NuScenes
        :param boxes: A list of boxes to generate ground truth for
        :type boxes: List
        :return: A generator that yields tuples containing the ground truth box, name, and instance token.
        :rtype: generator
        """
        from nuscenes.utils.data_classes import Box  # pylint: disable=import-error

        boxes: List[Box] = boxes

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)

        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        names = np.array([b.name for b in boxes])
        instance_tokens = [nuscenes.get("sample_annotation", box.token) for box in boxes]

        yield from zip(gt_boxes, names, instance_tokens)

    def convert_lidar_to_supervisely(self) -> str:
        """
        Converts a LiDAR point cloud file to the Supervisely format and saves it as a .pcd file.

        :return: The file path of the saved .pcd file.
        :rtype: str
        """
        import open3d as o3d  # pylint: disable=import-error

        bin_file = Path(self.lidar_path)
        save_path = str(bin_file.with_suffix(".pcd"))

        b = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
        points = b[:, 0:3]
        intensity = b[:, 3]
        ring_index = b[:, 4]
        intensity_fake_rgb = np.zeros((intensity.shape[0], 3))
        intensity_fake_rgb[:, 0] = (
            intensity  # red The intensity measures the reflectivity of the objects
        )
        intensity_fake_rgb[:, 1] = (
            ring_index  # green ring index is the index of the laser ranging from 0 to 31
        )
        try:
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            pc.colors = o3d.utility.Vector3dVector(intensity_fake_rgb)
            o3d.io.write_point_cloud(save_path, pc)
        except Exception as e:
            logger.warning(f"Error converting lidar to supervisely format: {e}")
        return save_path
