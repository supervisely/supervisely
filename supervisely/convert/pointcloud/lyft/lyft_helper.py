import os
from typing import List, Tuple
import numpy as np
from supervisely import logger, fs
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d

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
FOLDER_NAMES = ["data", "lidar", "images", "maps"]


def get_available_scenes(lyft):
    for scene in lyft.scene:
        token = scene["token"]
        scene_rec = lyft.get("scene", token)
        sample_rec = lyft.get("sample", scene_rec["first_sample_token"])
        sample_data = lyft.get("sample_data", sample_rec["data"]["LIDAR_TOP"])

        lidar_path, boxes, _ = lyft.get_sample_data(sample_data["token"])
        if not os.path.exists(str(lidar_path)):
            continue
        yield scene


def extract_data_from_scene(lyft, scene):
    try:
        from pyquaternion import Quaternion
        from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
    except ImportError as ie:
        logger.warn(f"Lazy import failed. Error: {ie}")
        return

    new_token = scene["first_sample_token"]
    dataset_data = []
    num_samples = scene["nbr_samples"]
    for i in range(num_samples):
        my_sample = lyft.get("sample", new_token)

        data = {}
        data["ann_data"] = {}
        data["ann_data"]["timestamp"] = my_sample["timestamp"]
        data["custom_data"] = lyft.get("log", scene["log_token"])

        sensor_token = my_sample["data"]["LIDAR_TOP"]
        lidar_path, boxes, _ = lyft.get_sample_data(sensor_token)
        if not os.path.exists(str(lidar_path)):
            logger.debug(f"Skipping sample {new_token} - lidar file doesn't exist")
            continue
        data["lidar_path"] = str(lidar_path)

        sd_record_lid = lyft.get("sample_data", sensor_token)
        cs_record_lid = lyft.get("calibrated_sensor", sd_record_lid["calibrated_sensor_token"])
        ego_record_lid = lyft.get("ego_pose", sd_record_lid["ego_pose_token"])

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)

        gt_boxes = np.concatenate([locs, dims, -rots + np.pi / 2], axis=1)
        names = np.array([b.name for b in boxes])
        data["ann_data"]["names"] = names
        data["ann_data"]["gt_boxes"] = gt_boxes
        instance_tokens = [lyft.get("sample_annotation", box.token) for box in boxes]
        data["ann_data"]["instance_tokens"] = instance_tokens

        for sensor, sensor_token in my_sample["data"].items():
            if "CAM" in sensor:
                img_path, boxes, cam_intrinsic = lyft.get_sample_data(sensor_token)
                if not os.path.exists(img_path):
                    logger.debug(f"pass {sensor} - image doesn't exist")
                    continue
                data["ann_data"][sensor] = str(img_path)

                sd_record_cam = lyft.get("sample_data", sensor_token)
                cs_record_cam = lyft.get(
                    "calibrated_sensor", sd_record_cam["calibrated_sensor_token"]
                )
                ego_record_cam = lyft.get("ego_pose", sd_record_cam["ego_pose_token"])
                cam_height = sd_record_cam["height"]
                cam_width = sd_record_cam["width"]

                lid_to_ego = transform_matrix(
                    cs_record_lid["translation"],
                    Quaternion(cs_record_lid["rotation"]),
                    inverse=False,
                )
                lid_ego_to_world = transform_matrix(
                    ego_record_lid["translation"],
                    Quaternion(ego_record_lid["rotation"]),
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

                data["ann_data"][f"{sensor}_extrinsic"] = np.hstack(
                    (velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1))
                )
                data["ann_data"][f"{sensor}_intrinsic"] = np.asarray(
                    cs_record_cam["camera_intrinsic"]
                )
                data["ann_data"][f"{sensor}_imsize"] = (cam_width, cam_height)
            else:
                logger.debug(f"pass {sensor} - isn't a camera")
        dataset_data.append(data)

        new_token = my_sample.get("next")
        if not new_token:
            break

    return dataset_data


def generate_rimage_infos(related_images: List[Tuple[str, str]], ann_data):
    sensors_to_skip = ["_intrinsic", "_extrinsic", "_imsize"]
    for sensor, image_path in related_images:
        if not any([sensor.endswith(s) for s in sensors_to_skip]):
            image_name = fs.get_file_name_with_ext(image_path)
            sly_path_img = os.path.join(os.path.dirname(image_path), image_name)
            img_info = {
                "name": image_name,
                "meta": {
                    "deviceId": sensor,
                    "timestamp": ann_data["timestamp"],
                    "sensorsData": {
                        "extrinsicMatrix": list(
                            ann_data[f"{sensor}_extrinsic"].flatten().astype(float)
                        ),
                        "intrinsicMatrix": list(
                            ann_data[f"{sensor}_intrinsic"].flatten().astype(float)
                        ),
                    },
                },
            }
            yield sly_path_img, img_info


def _convert_BEVBox3D_to_geometry(box):
    bbox = box.to_xyzwhlr()
    dim = bbox[[3, 5, 4]]
    pos = bbox[:3] + [0, 0, dim[1] / 2]
    yaw = bbox[-1]
    position = Vector3d(float(pos[0]), float(pos[1]), float(pos[2]))
    rotation = Vector3d(0, 0, float(-yaw))

    dimension = Vector3d(float(dim[0]), float(dim[2]), float(dim[1]))
    geometry = Cuboid3d(position, rotation, dimension)
    return geometry


def convert_bin_to_pcd(bin_file, save_filepath):
    import open3d as o3d  # pylint: disable=import-error

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
    o3d.io.write_point_cloud(save_filepath, pc)


def get_related_images(ann_data):
    # sensors_to_skip = ["_intrinsic", "_extrinsic", "_imsize"]
    # return [
    #     (sensor, img_path)
    #     for sensor, img_path in ann_data.items()
    #     if "CAM" in sensor and not any([sensor.endswith(s) for s in sensors_to_skip])
    # ]
    return [(sensor, img_path) for sensor, img_path in ann_data.items() if "CAM" in sensor]


def validate_ann_dir(ann_dir):
    if any([not fs.file_exists(f"{ann_dir}/{d}.json") for d in TABLE_NAMES]):
        return False
    return True


def lyft_annotation_to_BEVBox3D(data):
    import open3d as o3d  # pylint: disable=import-error

    boxes = data["gt_boxes"]
    names = data["names"]

    objects = []
    for name, box in zip(names, boxes):
        center = [float(box[0]), float(box[1]), float(box[2])]
        size = [float(box[3]), float(box[5]), float(box[4])]
        ry = float(box[6])

        yaw = ry - np.pi
        yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi
        world_cam = None
        objects.append(o3d.ml.datasets.utils.BEVBox3D(center, size, yaw, name, -1.0, world_cam))
        objects[-1].yaw = ry

    return objects
