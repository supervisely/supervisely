from typing import List
import numpy as np
from os import path as osp
from nuscenes.utils.data_classes import transform_matrix
from pyquaternion import Quaternion
import datetime
from supervisely import fs
from pathlib import Path
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d

class Sample:
    def __init__(self, timestamp, lidar_path, anns, cam_data):
        self.timestamp = datetime.utcfromtimestamp(self.timestamp / 1e6).isoformat()
        self.lidar_path = lidar_path
        self.anns = anns
        self.cam_data = cam_data

    @staticmethod
    def generate_boxes(nuscenes, boxes):
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)

        gt_boxes = np.concatenate([locs, dims, -rots + np.pi / 2], axis=1)
        names = np.array([b.name for b in boxes])
        instance_tokens = [nuscenes.get("sample_annotation", box.token) for box in boxes]

        yield from zip(gt_boxes, names, instance_tokens)
    
    def convert_lidar_to_supervisely(self):
        import open3d as o3d  # pylint: disable=import-error
        bin_file = Path(self.lidar_path)
        save_path = bin_file.with_suffix(".pcd")

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

        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pc.colors = o3d.utility.Vector3dVector(intensity_fake_rgb)
        o3d.io.write_point_cloud(save_path, pc)
        return save_path

class AnnotationObject:
    def __init__(self, name: str, bbox: np.ndarray, instance_token: List[str], parent_token: str, category: str, attributes: List[str], visibility: str):
        self.name = name
        self.bbox = bbox
        self.instance_token = instance_token
        self.parent_token = parent_token

        self.category = category
        self.attributes = attributes
        self.visibility = visibility

    def to_supervisely(self):
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
    def __init__(self, nuscenes, sensor_name, sensor_token, cs_record, ego_record):
        img_path, boxes, cam_intrinsic = nuscenes.get_sample_data(sensor_token)
        if not osp.exists(img_path):
            return None

        sd_record_cam = nuscenes.get("sample_data", sensor_token)
        cs_record_cam = nuscenes.get(
            "calibrated_sensor", sd_record_cam["calibrated_sensor_token"]
        )
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
        self.extrinsic = np.hstack(
            (velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1))
        )
        self.intrinsic = np.asarray(
            cs_record_cam["camera_intrinsic"]
        )

    def get_info(self, timestamp):
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
                        "extrinsicMatrix": list(
                            self.extrinsic.flatten().astype(float)
                        ),
                        "intrinsicMatrix": list(
                            self.intrinsic.flatten().astype(float)
                        ),
                    },
                },
            }
            return (sly_path_img, img_info)