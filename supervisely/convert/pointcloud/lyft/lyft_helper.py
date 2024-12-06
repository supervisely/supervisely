import os
import json
from typing import Dict, Any
import numpy as np
import open3d as o3d
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely import (
    PointcloudObject,
    PointcloudFigure,
    PointcloudAnnotation,
    PointcloudObjectCollection,
    dump_json_file,
    logger,
    Progress
)

from pyquaternion import Quaternion

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
        from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
    except ImportError:
        logger.error("Please run pip install lyft_dataset_sdk")
        return

    new_token = scene["first_sample_token"]
    my_sample = lyft.get('sample', new_token)

    dataset_data = []
    
    num_samples = scene["nbr_samples"] - 1 # TODO: fix, dont skip first frame
    progress = Progress("Extracting data from scene", num_samples)
    for i in range(num_samples):
        new_token = my_sample['next']
        my_sample = lyft.get('sample', new_token)

        data = {}
        data['timestamp'] = my_sample['timestamp']

        sensor_token = my_sample['data']['LIDAR_TOP']
        lidar_path, boxes, _ = lyft.get_sample_data(sensor_token)

        sd_record_lid = lyft.get("sample_data", sensor_token)
        cs_record_lid = lyft.get("calibrated_sensor", sd_record_lid["calibrated_sensor_token"])
        ego_record_lid = lyft.get("ego_pose", sd_record_lid["ego_pose_token"])

        assert os.path.exists(lidar_path)

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes
                         ]).reshape(-1, 1)

        names = np.array([b.name for b in boxes])
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        data['lidar_path'] = str(lidar_path)
        data["ann_data"]['names'] = names
        data["ann_data"]["gt_boxes"] = gt_boxes

        for sensor, sensor_token in my_sample['data'].items():
            if 'CAM' in sensor:
                img_path, boxes, cam_intrinsic = lyft.get_sample_data(sensor_token)
                assert os.path.exists(img_path)
                data["ann_data"][sensor] = str(img_path)

                sd_record_cam = lyft.get("sample_data", sensor_token)
                cs_record_cam = lyft.get("calibrated_sensor", sd_record_cam["calibrated_sensor_token"])
                ego_record_cam = lyft.get("ego_pose", sd_record_cam["ego_pose_token"])
                cam_height = sd_record_cam["height"]
                cam_width = sd_record_cam["width"]

                lid_to_ego = transform_matrix(
                    cs_record_lid["translation"], Quaternion(cs_record_lid["rotation"]), inverse=False
                )
                lid_ego_to_world = transform_matrix(
                    ego_record_lid["translation"], Quaternion(ego_record_lid["rotation"]), inverse=False
                )
                world_to_cam_ego = transform_matrix(
                    ego_record_cam["translation"], Quaternion(ego_record_cam["rotation"]), inverse=True
                )
                ego_to_cam = transform_matrix(
                    cs_record_cam["translation"], Quaternion(cs_record_cam["rotation"]), inverse=True
                )

                velo_to_cam = np.dot(ego_to_cam, np.dot(world_to_cam_ego, np.dot(lid_ego_to_world, lid_to_ego)))
                velo_to_cam_rot = velo_to_cam[:3, :3]
                velo_to_cam_trans = velo_to_cam[:3, 3]

                data["ann_data"][f'{sensor}_extrinsic'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
                data["ann_data"][f'{sensor}_intrinsic'] = np.asarray(cs_record_cam['camera_intrinsic'])
                data["ann_data"][f'{sensor}_imsize'] = (cam_width, cam_height)
            else:
                logger.debug(f"pass {sensor} - isn't a camera")
        progress.iter_done_report()
        dataset_data.append(data)
    return dataset_data