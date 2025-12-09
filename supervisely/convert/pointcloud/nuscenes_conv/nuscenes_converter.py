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

        tag_metas = [TagMeta(attr["name"], TagValueType.NONE) for attr in nuscenes.attribute]
        obj_classes = []
        classes_token_map = {}
        for category in nuscenes.category:
            color = nuscenes.colormap[category["name"]]
            description = helpers.trim_description(category["description"])
            obj_classes.append(ObjClass(category["name"], Cuboid3d, color, description=description))
            classes_token_map[category["token"]] = category["name"]
        self._custom_data["classes_token_map"] = classes_token_map

        self._meta = ProjectMeta(obj_classes, tag_metas)
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

            log = nuscenes.get("log", scene["log_token"])
            sample_token = scene["first_sample_token"]

            # * Extract scene's samples
            scene_samples: List[helpers.Sample] = []
            frame_token_map = {}
            for i in range(scene["nbr_samples"]):
                sample = nuscenes.get("sample", sample_token)
                frame_token_map[sample["token"]] = i
                lidar_path, boxes, _ = nuscenes.get_sample_data(sample["data"]["LIDAR_TOP"])
                if not os.path.exists(lidar_path):
                    logger.warning(f'Scene "{scene["name"]}" has no LIDAR data.')
                    continue

                timestamp = sample["timestamp"]
                anns = []
                for box, name, inst_token in helpers.Sample.generate_boxes(nuscenes, boxes):
                    current_instance_token = inst_token["token"]
                    parent_token = inst_token["prev"]

                    # get category, attributes and visibility
                    ann = nuscenes.get("sample_annotation", current_instance_token)
                    category = ann["category_name"]
                    attributes = [
                        nuscenes.get("attribute", attr)["name"] for attr in ann["attribute_tokens"]
                    ]
                    visibility = nuscenes.get("visibility", ann["visibility_token"])["level"]

                    ann_uuid = uuid.UUID(ann["token"])
                    ann = helpers.AnnotationObject(
                        name=name,
                        bbox=box,
                        token=ann_uuid,
                        instance_token=current_instance_token,
                        parent_token=parent_token,
                        category=category,
                        attributes=attributes,
                        visibility=visibility,
                    )
                    anns.append(ann)

                # get camera data
                sample_data = nuscenes.get("sample_data", sample["data"]["LIDAR_TOP"])
                cal_sensor = nuscenes.get(
                    "calibrated_sensor", sample_data["calibrated_sensor_token"]
                )
                ego_pose = nuscenes.get("ego_pose", sample_data["ego_pose_token"])

                camera_data = [
                    helpers.CamData(nuscenes, sensor, token, cal_sensor, ego_pose)
                    for sensor, token in sample["data"].items()
                    if sensor.startswith("CAM")
                ]
                scene_samples.append(helpers.Sample(timestamp, lidar_path, anns, camera_data))
                sample_token = sample["next"]
            self._custom_data["frame_token_map"][current_dataset_id] = frame_token_map

            # * Convert and upload pointclouds w/ annotations
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
