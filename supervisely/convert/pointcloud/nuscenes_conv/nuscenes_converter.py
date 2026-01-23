import os
import uuid
from typing import Dict, List, Optional

import supervisely.convert.pointcloud_episodes.nuscenes_conv.nuscenes_helper as helpers
import supervisely.io.fs as fs
from supervisely import KeyIdMap, PointcloudAnnotation, PointcloudObject
from supervisely._utils import is_development
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.api.api import Api, ApiField
from supervisely.api.dataset_api import DatasetInfo
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.convert.pointcloud_episodes.nuscenes_conv.nuscenes_converter import (
    NuscenesEpisodesConverter,
)
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)
from supervisely.pointcloud_annotation.pointcloud_tag import PointcloudTag
from supervisely.pointcloud_annotation.pointcloud_tag_collection import (
    PointcloudTagCollection,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger


class NuscenesConverter(NuscenesEpisodesConverter, PointcloudConverter):
    """Converter for NuScenes pointcloud format."""

    def to_supervisely(
        self,
        scene_sample: helpers.Sample,
        meta: ProjectMeta,
        renamed_classes: dict = {},
        renamed_tags: dict = {},
    ) -> PointcloudAnnotation:
        bevbox_objs = [obj.convert_nuscenes_to_BEVBox3D() for obj in scene_sample.anns]
        geoms = [obj.to_supervisely() for obj in scene_sample.anns]
        attrs = [obj.attributes for obj in scene_sample.anns]

        figures = []
        objs = []
        for label, geom, attributes in zip(bevbox_objs, geoms, attrs):
            class_name = renamed_classes.get(label.label_class, label.label_class)
            tag_col = None
            if len(attributes) > 0 and all([tag_name is not None for tag_name in attributes]):
                tag_meta_names = [renamed_tags.get(name, name) for name in attributes]
                tag_metas = [meta.get_tag_meta(tag_meta_name) for tag_meta_name in tag_meta_names]
                tag_col = PointcloudTagCollection([PointcloudTag(meta, None) for meta in tag_metas])
            pcobj = PointcloudObject(meta.get_obj_class(class_name), tag_col)
            figures.append(PointcloudFigure(pcobj, geom))
            objs.append(pcobj)
        return PointcloudAnnotation(PointcloudObjectCollection(objs), figures)

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        from nuscenes.nuscenes import NuScenes  # pylint: disable=import-error

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
            progress, progress_cb = self.get_progress(total_sample_cnt, "Converting pointclouds...")
        else:
            progress_cb = None

        self._custom_data["frame_token_map"] = {}
        for scene in nuscenes.scene:
            current_dataset_id = scene_name_to_dataset[scene["name"]].id

            # * Extract scene's samples
            parsed_samples = helpers.build_scene_samples(nuscenes, scene)
            scene_samples: List[helpers.Sample] = list(parsed_samples.values())
            frame_token_map = {k: i for i, k in enumerate(parsed_samples.keys())}
            self._custom_data["frame_token_map"][current_dataset_id] = frame_token_map

            # * Convert and upload pointclouds w/ annotations
            log = nuscenes.get("log", scene["log_token"])
            for idx, sample in enumerate(scene_samples):
                pcd_ann = self.to_supervisely(sample, meta, renamed_classes, renamed_tags)

                pcd_path = sample.convert_lidar_to_supervisely()
                pcd_name = fs.get_file_name_with_ext(pcd_path)
                pcd_meta = {
                    "frame": idx,
                    "vehicle": log["vehicle"],
                    "date": log["date_captured"],
                    "location": log["location"],
                    "description": scene["description"],
                }
                info = api.pointcloud.upload_path(current_dataset_id, pcd_name, pcd_path, pcd_meta)
                fs.silent_remove(pcd_path)

                pcd_id = info.id
                # * Upload pointcloud annotation
                try:
                    api.pointcloud.annotation.append(pcd_id, pcd_ann, key_id_map)
                except Exception as e:
                    error_msg = getattr(getattr(e, "response", e), "text", str(e))
                    logger.warning(
                        f"Failed to upload annotation for scene: {scene['name']}. Message: {error_msg}"
                    )

                # * Upload related images
                image_jsons = []
                camera_names = []
                for img_path, rimage_info in [
                    data.get_info(sample.timestamp) for data in sample.cam_data
                ]:
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

                if log_progress:
                    progress_cb(1)

            logger.info(f"Dataset ID:{current_dataset_id} has been successfully uploaded.")

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
