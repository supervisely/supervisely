from typing import Dict, Optional, Union

from supervisely.convert.base_converter import AvailablePointcloudEpisodesConverters
from supervisely.convert.pointcloud.lyft.lyft_converter import LyftConverter
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
from supervisely.pointcloud_annotation.pointcloud_episode_tag import (
    PointcloudEpisodeTag,
)
from supervisely.project.project_settings import LabelingInterface

from pathlib import Path
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
)
from supervisely.io import fs
from supervisely.convert.pointcloud.lyft import lyft_helper
from supervisely.api.api import ApiField
from datetime import datetime
from supervisely.geometry.cuboid_3d import Cuboid3d
from collections import defaultdict

# from supervisely.annotation.tag_meta import TagTargetType as TagTT


class LyftEpisodesConverter(LyftConverter, PointcloudEpisodeConverter):
    """Converter for LYFT pointcloud episodes format."""

    def __init__(
        self,
        input_data: str,
        labeling_interface: Optional[Union[LabelingInterface, str]],
        upload_as_links: bool,
        remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)

        self._type = "point_cloud_episode"
        self._is_pcd_episode = True

    def __str__(self) -> str:
        return AvailablePointcloudEpisodesConverters.LYFT

    def to_supervisely(
        self,
        items,
        meta: ProjectMeta,
        renamed_classes: dict = {},
        renamed_tags: dict = {},
    ):
        lyft = self._lyft

        scene_name_to_pcd_ep_ann = {}
        # * Group items by scene name
        scene_name_to_items = defaultdict(list)
        for item in items:
            scene_name_to_items[item._scene_name].append(item)

        # * Iterate over each scene
        for scene_name, items in scene_name_to_items.items():
            token_to_obj = {}
            frames = []
            tags = []  # todo tags that belong to the scene if any
            # * Iterate over each sample in the scene
            for i, item in enumerate(items):
                ann = item.ann_data
                objs = lyft_helper.lyft_annotation_to_BEVBox3D(ann)
                figures = []
                for obj, instance_token in zip(objs, ann["instance_tokens"]):
                    parent_object = None
                    parent_obj_token = instance_token["prev"]
                    if parent_obj_token == "":
                        # * Create a new object
                        class_name = instance_token["category_name"]
                        obj_class_name = renamed_classes.get(class_name, class_name)
                        obj_class = meta.get_obj_class(obj_class_name)

                        # * Get tags for the object
                        # tag_names = [
                        # lyft.get("attribute", attr_token).get("name", None)
                        # for attr_token in instance_token["attribute_tokens"]
                        # ]
                        # if len(tag_names) > 0 and all(
                        # [tag_name is not None for tag_name in tag_names]
                        # ):
                        # tags = [TagMeta(tag_name, TagValueType.NONE) for tag_name in tag_names]
                        # tag_meta_names = [renamed_tags.get(name, name) for name in tag_names]
                        # tag_metas = [
                        # meta.get_tag_meta(tag_meta_name) for tag_meta_name in tag_meta_names
                        # ]
                        # obj_tags = PointcloudEpisodeTagCollection(
                        # [PointcloudEpisodeTag(tag_meta, None) for tag_meta in tag_metas]
                        # )
                        obj_tags = None  # todo remove after fixing tags
                        pcd_ep_obj = PointcloudEpisodeObject(obj_class, obj_tags)
                        # * Assign the object to the starting token
                        token_to_obj[instance_token["token"]] = pcd_ep_obj
                        parent_object = pcd_ep_obj
                    else:
                        # * -> Figure has a parent object, get it
                        token_to_obj[instance_token["token"]] = token_to_obj[parent_obj_token]
                        parent_object = token_to_obj[parent_obj_token]

                    geom = lyft_helper._convert_BEVBox3D_to_geometry(obj)
                    pcd_figure = PointcloudFigure(parent_object, geom, i)
                    figures.append(pcd_figure)
                frame = PointcloudEpisodeFrame(i, figures)
                frames.append(frame)
            tag_collection = PointcloudEpisodeTagCollection(tags) if len(tags) > 0 else None
            pcd_ep_ann = PointcloudEpisodeAnnotation(
                len(frames),
                PointcloudEpisodeObjectCollection(list(set(token_to_obj.values()))),
                PointcloudEpisodeFrameCollection(frames),
                tag_collection,
            )
            scene_name_to_pcd_ep_ann[scene_name] = pcd_ep_ann
        return scene_name_to_pcd_ep_ann

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        unique_names = {name for item in self._items for name in item.ann_data["names"]}
        tag_names = {tag["name"] for tag in self._lyft.attribute}
        target_type = None  # TagTT.GLOBAL # todo remove after fixing tags
        self._meta = ProjectMeta(
            [ObjClass(name, Cuboid3d) for name in unique_names],
            [TagMeta(tag, TagValueType.NONE, target_type=target_type) for tag in tag_names],
        )
        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        scene_names = set([item._scene_name for item in self._items])

        dataset_info = api.dataset.get_info_by_id(dataset_id)
        scene_name_to_dataset = {}

        multiple_scenes = len(scene_names) > 1
        if multiple_scenes:
            logger.info(
                f"Found {len(scene_names)} scenes ({self.items_count} pointclouds) in the input data."
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
            progress, progress_cb = self.get_progress(self.items_count, "Converting pointclouds...")
        else:
            progress_cb = None

        scene_name_to_ann = self.to_supervisely(self._items, meta, renamed_classes, renamed_tags)
        scene_name_to_item = defaultdict(list)
        for item in self._items:
            scene_name_to_item[item._scene_name].append(item)

        for scene, items in scene_name_to_item.items():
            # * Get the annotation for the scene
            ann_episode = scene_name_to_ann[scene]
            current_dataset_id = scene_name_to_dataset[item._scene_name].id
            frame_to_pointcloud_ids = {}
            for idx, item in enumerate(items):
                # * Convert timestamp to ISO format
                iso_time = (
                    datetime.utcfromtimestamp(item.ann_data["timestamp"] / 1e6).isoformat() + "Z"
                )
                item.ann_data["timestamp"] = iso_time

                # * Convert pointcloud from ".bin" to ".pcd"
                pcd_path = str(Path(item.path).with_suffix(".pcd"))
                if fs.file_exists(pcd_path):
                    logger.warning(f"Overwriting file with path: {pcd_path}")
                lyft_helper.convert_bin_to_pcd(item.path, pcd_path)

                # * Upload pointcloud
                pcd_meta = {}
                pcd_meta["frame"] = idx

                pcd_name = fs.get_file_name(pcd_path)
                info = api.pointcloud_episode.upload_path(
                    current_dataset_id, pcd_name, pcd_path, pcd_meta
                )
                pcd_id = info.id
                frame_to_pointcloud_ids[idx] = pcd_id

                # * Upload related images
                image_jsons = []
                camera_names = []
                for img_path, rimage_info in lyft_helper.generate_rimage_infos(
                    item._related_images, item.ann_data
                ):
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

                # * Clean up
                fs.silent_remove(pcd_path)
                if log_progress:
                    progress_cb(1)

            try:
                api.pointcloud_episode.annotation.append(
                    current_dataset_id, ann_episode, frame_to_pointcloud_ids
                )
            except Exception as e:
                error_msg = getattr(getattr(e, "response", e), "text", str(e))
                logger.warn(f"Failed to upload annotation for scene: {scene}. Message: {error_msg}")
            logger.info(f"Dataset ID:{current_dataset_id} has been successfully uploaded.")

        if log_progress:
            if is_development():
                progress.close()
