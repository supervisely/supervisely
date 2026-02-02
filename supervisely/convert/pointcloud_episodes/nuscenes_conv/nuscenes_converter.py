from __future__ import annotations

import os
import uuid
from os import path as osp
from pathlib import Path
from typing import Dict, List, Optional

import supervisely.convert.pointcloud_episodes.nuscenes_conv.nuscenes_helper as helpers
import supervisely.io.fs as fs
from supervisely._utils import is_development
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.api.api import Api, ApiField
from supervisely.api.dataset_api import DatasetInfo
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import (
    PointcloudEpisodeConverter,
)
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.pointcloud_annotation.pointcloud_episode_annotation import (
    PointcloudEpisodeAnnotation,
)
from supervisely.pointcloud_annotation.pointcloud_episode_frame import (
    PointcloudEpisodeFrame,
)
from supervisely.pointcloud_annotation.pointcloud_episode_frame_collection import (
    PointcloudEpisodeFrameCollection,
)
from supervisely.pointcloud_annotation.pointcloud_episode_object import (
    PointcloudEpisodeObject,
)
from supervisely.pointcloud_annotation.pointcloud_episode_object_collection import (
    PointcloudEpisodeObjectCollection,
)
from supervisely.pointcloud_annotation.pointcloud_episode_tag_collection import (
    PointcloudEpisodeTagCollection,
)
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.tiny_timer import TinyTimer
from supervisely.video_annotation.key_id_map import KeyIdMap


class NuscenesEpisodesConverter(PointcloudEpisodeConverter):
    """Converter for NuScenes pointcloud episodes format."""

    _nuscenes: "NuScenes" = None  # type: ignore
    _custom_data: Dict = {}

    def __str__(self) -> str:
        return AvailablePointcloudConverters.NUSCENES

    def validate_format(self) -> bool:
        try:
            from nuscenes import NuScenes
        except ImportError:
            logger.warning("Please, run 'pip install nuscenes-devkit' to import NuScenes data.")
            return False

        table_json_filenames = [f"{name}.json" for name in helpers.TABLE_NAMES]

        def _contains_tables(dir_path: str) -> bool:
            return all(fs.file_exists(osp.join(dir_path, table)) for table in table_json_filenames)

        def _filter_fn(path):
            has_tables = False
            for p in os.scandir(path):
                if p.is_dir() and _contains_tables(p.path):
                    has_tables = True
                    break
            return has_tables and (Path(path) / "samples").exists()

        input_path = next((d for d in fs.dirs_filter(self._input_data, _filter_fn)), None)
        if input_path is None:
            return False

        ann_dir = next((d for d in fs.dirs_filter(input_path, _contains_tables)), None)
        if ann_dir is None:
            return False

        version = osp.basename(ann_dir)
        try:
            t = TinyTimer()
            self._nuscenes: NuScenes = NuScenes(version=version, dataroot=input_path, verbose=False)
            logger.debug(f"NuScenes initialization took {t.get_sec():.3f} sec")
        except Exception as e:
            logger.debug(f"Failed to initialize NuScenes: {e}")
            return False

        self._custom_data["nuscenes_version"] = version
        self._custom_data["dataroot"] = input_path
        return True

    def to_supervisely(
        self,
        scene_samples: Dict[str, helpers.Sample],
        meta: ProjectMeta,
        renamed_classes: dict = {},
        renamed_tags: dict = {},
    ) -> PointcloudEpisodeAnnotation:
        token_to_obj = {}
        frames = []
        tags = []
        frame_idx_to_scene_sample_token = {}
        if "frame_token_map" not in self._custom_data:
            self._custom_data["frame_token_map"] = {}
        for sample_i, (token, sample) in enumerate(scene_samples.items()):
            figures = []
            for obj in sample.anns:
                ann_token = uuid.UUID(obj.token)
                instance_token = obj.instance_token
                class_name = obj.category
                parent_obj_token = obj.parent_token
                parent_object = None
                try:
                    if parent_obj_token == "":
                        # * Create a new object
                        obj_class_name = renamed_classes.get(class_name, class_name)
                        obj_class = meta.get_obj_class(obj_class_name)
                        obj_tags = None  # ! TODO: fix tags
                        pcd_ep_obj = PointcloudEpisodeObject(
                            obj_class, obj_tags, uuid.UUID(instance_token)
                        )
                        # * Assign the object to the starting token
                        token_to_obj[instance_token] = pcd_ep_obj
                        parent_object = pcd_ep_obj
                    else:
                        # * -> Figure has a parent object, get it
                        token_to_obj[instance_token] = token_to_obj[parent_obj_token]
                        parent_object = token_to_obj[parent_obj_token]
                    geom = obj.to_supervisely()
                    pcd_figure = PointcloudFigure(parent_object, geom, sample_i, ann_token)
                except Exception as e:
                    logger.error(
                        f"Error during conversion of object with token {ann_token}: {repr(e)}"
                    )
                    continue
                figures.append(pcd_figure)
                frame_idx_to_scene_sample_token[sample_i] = token
            frame = PointcloudEpisodeFrame(sample_i, figures)
            frames.append(frame)
        tag_collection = PointcloudEpisodeTagCollection(tags) if len(tags) > 0 else None
        self._custom_data["frame_token_map"][self._current_ds_id] = frame_idx_to_scene_sample_token
        key_uuid = uuid.UUID(token)
        return PointcloudEpisodeAnnotation(
            len(frames),
            PointcloudEpisodeObjectCollection(list(set(token_to_obj.values()))),
            PointcloudEpisodeFrameCollection(frames),
            tag_collection,
            key=key_uuid,
        )

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        from nuscenes import NuScenes  # pylint: disable=import-error

        nuscenes: NuScenes = self._nuscenes
        key_id_map = KeyIdMap()

        project_meta, classes_token_map = helpers.build_project_meta(nuscenes)
        self._custom_data["classes_token_map"] = classes_token_map

        self._meta = project_meta
        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        dataset_info = api.dataset.get_info_by_id(dataset_id)
        scene_name_to_dataset: Dict[str, DatasetInfo] = {}

        scene_names = [scene["name"] for scene in nuscenes.scene]
        scene_cnt = len(scene_names)
        total_sample_cnt = sum([scene["nbr_samples"] for scene in nuscenes.scene])

        multiple_scenes = len(scene_names) > 1
        if multiple_scenes:
            logger.info(f"Found {scene_cnt} scenes ({total_sample_cnt} samples) in the input data.")
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
            scene_name_to_dataset[scene_names[0]] = dataset_info

        if log_progress:
            progress, progress_cb = self.get_progress(
                total_sample_cnt, "Converting episode scenes..."
            )
        else:
            progress_cb = None

        for scene in nuscenes.scene:
            current_dataset_id = scene_name_to_dataset[scene["name"]].id
            self._current_ds_id = current_dataset_id

            log = nuscenes.get("log", scene["log_token"])

            # * Extract scene's samples
            scene_samples: Dict[str, helpers.Sample] = helpers.build_scene_samples(nuscenes, scene)

            # * Convert and upload pointclouds
            frame_to_pointcloud_ids = {}
            for idx, sample in enumerate(scene_samples.values()):
                pcd_path = sample.convert_lidar_to_supervisely()

                pcd_name = fs.get_file_name_with_ext(pcd_path)
                pcd_meta = {
                    "frame": idx,
                    "vehicle": log["vehicle"],
                    "date": log["date_captured"],
                    "location": log["location"],
                    "description": scene["description"],
                }
                info = api.pointcloud_episode.upload_path(
                    current_dataset_id, pcd_name, pcd_path, pcd_meta
                )
                fs.silent_remove(pcd_path)

                pcd_id = info.id
                frame_to_pointcloud_ids[idx] = pcd_id

                # * Upload related images
                image_jsons = []
                camera_names = []
                for img_path, rimage_info in [
                    data.get_info(sample.timestamp) for data in sample.cam_data
                ]:
                    img = api.pointcloud_episode.upload_related_image(img_path)
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
                    api.pointcloud_episode.add_related_images(image_jsons, camera_names)

                if log_progress:
                    progress_cb(1)

            # * Convert and upload annotations
            pcd_ann = self.to_supervisely(scene_samples, meta, renamed_classes, renamed_tags)

            try:
                api.pointcloud_episode.annotation.append(
                    current_dataset_id, pcd_ann, frame_to_pointcloud_ids, key_id_map=key_id_map
                )
                logger.info(f"Dataset ID:{current_dataset_id} has been successfully uploaded.")
            except Exception as e:
                error_msg = getattr(getattr(e, "response", e), "text", str(e))
                logger.warning(
                    f"Failed to upload annotation for scene: {scene['name']}. Message: {error_msg}"
                )
        key_id_map = key_id_map.to_dict()
        key_id_map.pop("tags")
        key_id_map.pop("videos")
        self._custom_data["key_id_map"] = key_id_map

        project_id = dataset_info.project_id
        current_custom_data = api.project.get_custom_data(project_id)
        current_custom_data.update(self._custom_data)
        api.project.update_custom_data(project_id, current_custom_data)

        if log_progress:
            if is_development():
                progress.close()
