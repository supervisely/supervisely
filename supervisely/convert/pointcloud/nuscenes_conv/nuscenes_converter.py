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
from supervisely.convert.pointcloud.lyft import lyft_helper
from supervisely.api.api import ApiField
from datetime import datetime
from supervisely import TinyTimer
from supervisely.pointcloud_annotation.pointcloud_annotation import (
    PointcloudFigure,
    PointcloudObjectCollection,
    PointcloudTagCollection,
)
from supervisely.pointcloud_annotation.pointcloud_tag import PointcloudTag

from supervisely.convert.pointcloud_episodes.nuscenes_conv.nuscenes_converter import (
    NuscenesEpisodesConverter,
)
from supervisely.convert.pointcloud_episodes.nuscenes_conv.nuscenes_helper import (
    Sample,
    AnnotationObject,
    CamData,
    TABLE_NAMES,
    DIR_NAMES,
)
from os import path as osp


class NuscenesConverter(NuscenesEpisodesConverter, PointcloudConverter):
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
        return AvailablePointcloudConverters.LYFT

    def to_supervisely(
        self,
        scene_sample,
        meta: ProjectMeta,
        renamed_classes: dict = {},
        renamed_tags: dict = {},
    ) -> PointcloudAnnotation:
        bevbox_objs = [obj.convert_nuscenes_to_BEVBox3D() for obj in scene_sample.anns]
        geoms = [obj.to_supervisely() for obj in scene_sample.anns]
        tokens = [obj.instance_token for obj in scene_sample.anns]

        figures = []
        objs = []
        for l, g, t in zip(bevbox_objs, geoms, tokens):
            class_name = renamed_classes.get(l.label_class, l.label_class)
            tag_names = [
                self._nuscenes.get("attribute", attr_token).get("name", None)
                for attr_token in t["attribute_tokens"]
            ]
            tag_col = None
            if len(tag_names) > 0 and all([tag_name is not None for tag_name in tag_names]):
                tag_meta_names = [renamed_tags.get(name, name) for name in tag_names]
                tag_metas = [meta.get_tag_meta(tag_meta_name) for tag_meta_name in tag_meta_names]
                tag_col = PointcloudTagCollection([PointcloudTag(meta, None) for meta in tag_metas])
            pcobj = PointcloudObject(meta.get_obj_class(class_name), tag_col)
            figures.append(PointcloudFigure(pcobj, g))
            objs.append(pcobj)
        return PointcloudAnnotation(PointcloudObjectCollection(objs), figures)

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        nuscenes = self._nuscenes

        unique_names = {name for item in self._items for name in item.ann_data["names"]}
        tag_names = {tag["name"] for tag in self._lyft.attribute}
        self._meta = ProjectMeta(
            [ObjClass(name, Cuboid3d) for name in unique_names],
            [TagMeta(tag, TagValueType.NONE) for tag in tag_names],
        )
        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        dataset_info = api.dataset.get_info_by_id(dataset_id)
        scene_name_to_dataset = {}

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
            progress, progress_cb = self.get_progress(self.items_count, "Converting pointclouds...")
        else:
            progress_cb = None

        for scene in nuscenes.scene:
            current_dataset_id = scene_name_to_dataset[scene["name"]].id
            sample = nuscenes.get("sample", scene["first_sample_token"])
            lidar_path, boxes, _ = nuscenes.get_sample_data(sample["data"]["LIDAR_TOP"])
            if not osp.exists(lidar_path):
                continue

            log = nuscenes.get("log", scene["log_token"])

            # todo various log data, to store in tags/meta
            vechicle = log["vehicle"]
            date = log["date_captured"]
            loc = log["location"]
            desc = scene["description"]

            scene_samples = []
            for i in range(scene["nbr_samples"]):
                timestamp = sample["timestamp"]
                anns = []
                for box, name, inst_token in Sample.generate_boxes(nuscenes, boxes):
                    current_instance_token = inst_token["token"]
                    parent_token = inst_token["prev"]

                    # get category, attributes and visibility
                    ann = nuscenes.get("sample_annotation", current_instance_token)
                    category = ann["category_name"]
                    attributes = [
                        nuscenes.get("attribute", attr)["name"] for attr in ann["attribute_tokens"]
                    ]
                    visibility = nuscenes.get("visibility", ann["visibility_token"])["level"]

                    anns.append(
                        AnnotationObject(
                            name,
                            box,
                            current_instance_token,
                            parent_token,
                            category,
                            attributes,
                            visibility,
                        )
                    )

                # get camera data
                sample_data = nuscenes.get("sample_data", sample["data"]["LIDAR_TOP"])
                cal_sensor = nuscenes.get(
                    "calibrated_sensor", sample_data["calibrated_sensor_token"]
                )
                ego_pose = nuscenes.get("ego_pose", sample_data["ego_pose_token"])

                camera_data = [
                    CamData(nuscenes, sensor, token, cal_sensor, ego_pose)
                    for sensor, token in sample["data"].items()
                    if sensor.startswith("CAM")
                ]
                scene_samples.append(Sample(timestamp, lidar_path, anns, camera_data))
                sample = nuscenes.get("sample", sample["next"])

            # * Convert and upload pointclouds w/ annotations
            for idx, sample in enumerate(scene_samples):
                pcd_ann = self.to_supervisely(sample, meta, renamed_classes, renamed_tags)

                pcd_path = sample.convert_lidar_to_supervisely()
                pcd_name = fs.get_file_name(pcd_path)
                pcd_meta = {}
                pcd_meta["frame"] = idx
                info = api.pointcloud.upload_path(current_dataset_id, pcd_name, pcd_path, pcd_meta)
                fs.silent_remove(pcd_path)

                pcd_id = info.id
                # * Upload pointcloud annotation
                try:
                    api.pointcloud.annotation.append(pcd_id, pcd_ann, {idx: pcd_id})
                except Exception as e:
                    error_msg = getattr(getattr(e, "response", e), "text", str(e))
                    logger.warn(
                        f"Failed to upload annotation for scene: {scene}. Message: {error_msg}"
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

            logger.info(f"Dataset ID:{current_dataset_id} has been successfully uploaded.")

            if log_progress:
                progress_cb(1)

        if log_progress:
            if is_development():
                progress.close()
