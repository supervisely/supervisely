import numpy as np
from pathlib import Path
from typing import Dict, Optional

from supervisely import (
    Api,
    ObjClass,
    PointcloudAnnotation,
    ProjectMeta,
    logger,
    is_development,
    Progress,
    PointcloudObject,
    TagMeta,
    TagValueType,
)
from supervisely.io import fs
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.convert.pointcloud_episodes.nuscenes.nuscenes_helper import Sample, AnnotationObject, CamData
from supervisely.api.api import ApiField
from datetime import datetime
from supervisely import TinyTimer
from supervisely.pointcloud_annotation.pointcloud_annotation import (
    PointcloudFigure,
    PointcloudObjectCollection,
    PointcloudTagCollection,
)
from supervisely.pointcloud_annotation.pointcloud_tag import PointcloudTag
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import (
    PointcloudEpisodeConverter,
)
from supervisely.pointcloud_annotation.pointcloud_episode_annotation import (
    PointcloudEpisodeAnnotation,
    PointcloudEpisodeObjectCollection,
    PointcloudEpisodeFrameCollection,
    PointcloudEpisodeTagCollection,
    PointcloudFigure,
)
from os import path as osp
from supervisely import (
    Api,
    ObjClass,
    ProjectMeta,
    logger,
    is_development,
    PointcloudObject,
    PointcloudEpisodeObject,
    PointcloudFigure,
    PointcloudEpisodeFrame,
    TagMeta,
    TagValueType,
    VideoTagCollection,
    VideoTag,
)

class NuscenesEpisodesConverter(PointcloudEpisodeConverter):
    class Item(PointcloudEpisodeConverter.Item):
        def __init__(
            self,
            item_path,
            ann_data: str = None,
            related_images: list = None,
            custom_data: dict = None,
            scene_name: str = None,
        ):
            super().__init__(item_path, ann_data, related_images, custom_data)
            self._type = "point_cloud"
            self._scene_name = scene_name

    def __init__(
        self,
        input_data: str,
        labeling_interface: str,
        upload_as_links: bool,
        remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)
        self._nuscenes = None

    def __str__(self) -> str:
        return AvailablePointcloudConverters.NUSCENES

    # @property
    # def key_file_ext(self) -> str:
    #     return ".bin"

    def validate_format(self) -> bool:
        try:
            from nuscenes import NuScenes
        except ImportError:
            return False
        
        # TODO: implement nuscenes format validation

        # def filter_fn(path):
        #     return all([(Path(path) / name).exists() for name in lyft_helper.FOLDER_NAMES])

        # input_paths = [d for d in fs.dirs_filter(self._input_data, filter_fn)]
        # if len(input_paths) == 0:
            # return False
        # input_path = input_paths[0]

        # lidar_dir = input_path + "/lidar/"
        # json_dir = input_path + "/data/"
        # if lyft_helper.validate_ann_dir(json_dir) is False:
        #     return False

        # bin_files = fs.list_files_recursively(
            # lidar_dir, [self.key_file_ext], ignore_valid_extensions_case=True
        # )
# 
        # if len(bin_files) == 0:
            # return False
# 
        # check if pointclouds have 5 columns (x, y, z, intensity, ring)
        # pointcloud = np.fromfile(bin_files[0], dtype=np.float32)
        # if pointcloud.shape[0] % 5 != 0:
            # return False
        nuscenes_version, root = None, None
        try:
            t = TinyTimer()
            nuscenes = NuScenes(nuscenes_version, root)
            self._nuscenes: NuScenes = nuscenes
            logger.info(f"NuScenes initialization took {t.get_sec():.2f} sec")
        except Exception as e:
            logger.info(f"Failed to initialize NuScenes: {e}")
            return False

        # t = TinyTimer()
        # progress = Progress(f"Extracting annotations from available scenes...")
        # # i = 0 # for debug
        # for scene in lyft_helper.get_available_scenes(lyft):
        #     scene_name = scene["name"]
        #     sample_datas = lyft_helper.extract_data_from_scene(lyft, scene)
        #     if sample_datas is None:
        #         logger.warning(f"Failed to extract sample data from scene: {scene['name']}.")
        #         continue
        #     for sample_data in sample_datas:
        #         item_path = sample_data["lidar_path"]
        #         ann_data = sample_data["ann_data"]
        #         related_images = lyft_helper.get_related_images(ann_data)
        #         custom_data = sample_data.get("custom_data", {})
        #         item = self.Item(item_path, ann_data, related_images, custom_data, scene_name)
        #         self._items.append(item)
        #     progress.iter_done_report()
        #     # i += 1
        #     # if i == 2:
        #     #     break
        # t = t.get_sec()
        # logger.info(
        #     f"Lyft annotation extraction took {t:.2f} sec ({(t / self.items_count):.3f} sec per sample)"
        # )

        return True

    def to_supervisely(
        self,
        scene_samples,
        meta: ProjectMeta,
        renamed_classes: dict = {},
        renamed_tags: dict = {},
    ) -> PointcloudAnnotation:
        token_to_obj = {}
        frames = []
        tags = []
        for sample_i, sample in enumerate(scene_samples):
            figures = []
            for i, obj in enumerate(sample.anns):
                instance_token = obj.instance_token
                class_name = obj.category
                parent_obj_token = obj.parent_token
                parent_object = None
                if parent_obj_token == "":
                    # * Create a new object
                    obj_class_name = renamed_classes.get(class_name, class_name)
                    obj_class = meta.get_obj_class(obj_class_name)
                    obj_tags = None # ! TODO: fix tags 
                    pcd_ep_obj = PointcloudEpisodeObject(obj_class, obj_tags)
                    # * Assign the object to the starting token
                    token_to_obj[instance_token] = pcd_ep_obj
                    parent_object = pcd_ep_obj
                else:
                    # * -> Figure has a parent object, get it
                    token_to_obj[instance_token] = token_to_obj[parent_obj_token]
                    parent_object = token_to_obj[parent_obj_token]
                geom = obj.to_supervisely()
                pcd_figure = PointcloudFigure(parent_object, geom, i)
                figures.append(pcd_figure)
            frame = PointcloudEpisodeFrame(sample_i, figures)
            frames.append(frame)
        tag_collection = PointcloudEpisodeTagCollection(tags) if len(tags) > 0 else None
        return PointcloudEpisodeAnnotation(
            len(frames),
            PointcloudEpisodeObjectCollection(list(set(token_to_obj.values()))),
            PointcloudEpisodeFrameCollection(frames),
            tag_collection,
        )

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        nuscenes = self._nuscenes
        attributes = {attr["name"]: attr["description"] for attr in nuscenes.attribute}
        category_to_color = {category['name']: nuscenes.colormap[category['name']] for category in nuscenes.category}
        self._meta = ProjectMeta(
            [ObjClass(name, Cuboid3d, color) for name, color in category_to_color.items()],
            [TagMeta(tag, TagValueType.NONE) for tag in attributes.keys()],
        )
        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        # scene_names = set([item._scene_name for item in self._items])
        dataset_info = api.dataset.get_info_by_id(dataset_id)
        scene_name_to_dataset = {}

        scene_names = None
        scene_cnt = len(scene_names)
        multiple_scenes = len(scene_names) > 1
        if multiple_scenes:
            logger.info(
                f"Found {len(scene_names)} scenes ({scene_cnt} pointclouds) in the input data."
            )
            # * Create a nested dataset for each scene
            for name in scene_names:
                ds = api.dataset.create(
                    dataset_info.project_id,
                    name,
                    change_name_if_conflict=True,
                    parent_id=dataset_id,
                )
                scene_name_to_dataset[name] = ds
        else:
            scene_name_to_dataset[list(scene_names)[0]] = dataset_info

        if log_progress:
            progress, progress_cb = self.get_progress(scene_cnt, "Converting pointclouds...")
        else:
            progress_cb = None

        for scene in nuscenes.scene:
            current_dataset_id = scene_name_to_dataset[scene['name']].id
            sample = nuscenes.get('sample', scene['first_sample_token'])
            lidar_path, boxes, _ = nuscenes.get_sample_data(sample["data"]["LIDAR_TOP"])
            if not osp.exists(lidar_path):
                continue

            # gather log info
            log = nuscenes.get('log', scene['log_token'])

            # todo various log data, to store in tags/meta
            vechicle = log['vehicle']
            date = log['date_captured']
            loc = log['location']
            desc = scene['description']

            scene_samples = []
            while True:
                timestamp = sample['timestamp']
                anns = []
                for box, name, inst_token in Sample.generate_boxes(nuscenes, boxes):
                    current_instance_token = inst_token['token']
                    parent_token = inst_token['prev']

                    # get category, attributes and visibility
                    ann = nuscenes.get("sample_annotation", current_instance_token)
                    category = ann['category_name']
                    attributes = [nuscenes.get('attribute', attr)['name'] for attr in ann['attribute_tokens']]
                    visibility = nuscenes.get("visibility", ann['visibility_token'])['level']

                    anns.append(AnnotationObject(name, box, current_instance_token, parent_token, category, attributes, visibility))

                # get camera data
                sample_data = nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])
                cal_sensor = nuscenes.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                ego_pose = nuscenes.get('ego_pose', sample_data['ego_pose_token'])

                camera_data = [CamData(nuscenes, sensor, token, cal_sensor, ego_pose) for sensor, token in sample['data'].items() if sensor.startswith('CAM')]
                scene_samples.append(Sample(timestamp, lidar_path, anns, camera_data))
                if sample['next'] == '':
                    break
                sample = nuscenes.get('sample', sample['next'])

            # * Convert and upload pointclouds
            frame_to_pointcloud_ids = {}
            for idx, sample in enumerate(scene_samples):
                pcd_path = sample.convert_lidar_to_supervisely()
                pcd_name = fs.get_file_name(pcd_path)
                pcd_meta = {}
                pcd_meta["frame"] = idx
                info = api.pointcloud_episode.upload_path(
                    current_dataset_id, pcd_name, pcd_path, pcd_meta
                )
                pcd_id = info.id
                frame_to_pointcloud_ids[idx] = pcd_id
                fs.silent_remove(pcd_path)

                # * Upload related images
                image_jsons = []
                camera_names = []
                for img_path, rimage_info in [data.get_info(sample.timestamp) for data in sample.cam_data]:
                    img = api.pointcloud.upload_related_image(img_path)
                    image_jsons.append(
                        {
                            ApiField.ENTITY_ID: pcd_id,
                            ApiField.NAME: rimage_info[ApiField.NAME],
                            ApiField.HASH: img,
                            ApiField.META: rimage_info[ApiField.META],
                        }
                    )
                    camera_names.append(rimage_info[ApiField.META]["deviceId"])
                if len(image_jsons) > 0:
                    api.pointcloud.add_related_images(image_jsons, camera_names)

            # * Convert and upload annotations
            pcd_ann = self.to_supervisely(scene_samples, meta, renamed_classes, renamed_tags)
            try:
                api.pointcloud_episode.annotation.append(
                    current_dataset_id, pcd_ann, frame_to_pointcloud_ids
                )
            except Exception as e:
                error_msg = getattr(getattr(e, "response", e), "text", str(e))
                logger.warn(
                    f"Failed to upload annotation for scene: {scene}. Message: {error_msg}"
                )
            logger.info(f"Dataset ID:{current_dataset_id} has been successfully uploaded.")

            if log_progress:
                progress_cb(1)


        if log_progress:
            if is_development():
                progress.close()
