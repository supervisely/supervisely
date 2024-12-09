import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
from collections import defaultdict

from supervisely import (
    Api,
    ObjClass,
    PointcloudAnnotation,
    PointcloudEpisodeAnnotation,
    PointcloudEpisodeFrame,
    PointcloudEpisodeObject,
    PointcloudFigure,
    ProjectMeta,
    generate_free_name,
    logger,
    is_development,
)
from supervisely.io.fs import silent_remove
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely.io.fs import get_file_ext, get_file_name, list_files_recursively, silent_remove
from supervisely.io.json import load_json_file
from supervisely.convert.pointcloud.lyft import lyft_helper
from supervisely.api.api import ApiField
from datetime import datetime


class LyftConverter(PointcloudConverter):
    class Item(PointcloudConverter.Item):
        def __init__(
            self,
            item_path,
            ann_data: str = None,
            scene_name: str = None,
            related_images: list = None,
            custom_data: dict = None,
        ):
            super().__init__(item_path, ann_data, related_images, custom_data)
            self._type = "point_cloud"
            self.scene_name = scene_name

    def __init__(
        self,
        input_data: str,
        labeling_interface: str,
        upload_as_links: bool,
        remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)
        self._total_msg_count = 0
        self._is_pcd_episode = False
        self.meta_needs_update = False

    def __str__(self) -> str:
        return AvailablePointcloudConverters.LYFT

    @property
    def key_file_ext(self) -> str:
        return ".bin"

    def validate_format(self) -> bool:
        try:
            from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
        except ImportError:
            logger.error(
                'Please run "pip install lyft_dataset_sdk" ' "to install the official devkit first."
            )
            return

        def _filter_fn(file_path):
            return get_file_ext(file_path).lower() == self.key_file_ext

        bin_files = list_files_recursively(self._input_data, filter_fn=_filter_fn)

        if len(bin_files) == 0:
            return False
        else:
            # check if pointclouds have 5 columns (x, y, z, intensity, ring)
            pointcloud = np.fromfile(bin_files[0], dtype=np.float32)
            if pointcloud.shape[0] % 5 != 0:
                return False

        # @TODO: move code below to validate_ann_file method
        json_path = self._input_data + "/data/"  # todo: find the json folder
        lyft = Lyft(data_path=self._input_data, json_path=json_path, verbose=False)

        available_scenes = [scene for scene in lyft_helper.get_available_scenes(lyft)]
        for scene in available_scenes:
            sample_datas = lyft_helper.extract_data_from_scene(lyft, scene)
            if sample_datas is None:
                logger.warning(f"Failed to extract sample data from scene: {scene['name']}.")
                continue
            for sample_data in sample_datas:
                item_path = sample_data["lidar_path"]
                ann_data = sample_data["ann_data"]
                related_images = lyft_helper.get_related_images(ann_data)
                custom_data = sample_data.get("custom_data", {})  # todo: implement
                scene_name = scene["name"]
                item = self.Item(item_path, ann_data, scene_name, related_images, custom_data)
                self._items.append(item)
            break

        return self.items_count > 0

    def convert(self, item: PointcloudConverter.Item, meta: ProjectMeta):
        time_to_data = {}
        timestamp = item.ann_data["timestamp"]
        time = datetime.utcfromtimestamp(timestamp / 1e6).isoformat() + "Z"
        item.ann_data["timestamp"] = time
        time_to_data[time] = {}
        save_path = item.path[:-4] + ".pcd"
        lyft_helper.convert_bin_to_pcd(item.path, save_path)
        time_to_data[time]["pcd"] = save_path

        ann_path = item.path[:-4] + ".json"
        label = lyft_helper.lyft_annotation_to_BEVBox3D(item.ann_data)

        meta_class_names = [obj_class.name for obj_class in meta.obj_classes]
        classes_to_add = set()
        for l in label:
            if l.label_class not in meta_class_names:
                classes_to_add.add(l.label_class)

        if len(classes_to_add) > 0:
            meta = meta.add_obj_classes(
                [ObjClass(objclass, Cuboid3d) for objclass in classes_to_add]
            )
            self._meta = meta
            self.meta_needs_update = True
        lyft_helper.convert_label_to_annotation(label, ann_path, meta)  # to move, causes issues
        time_to_data[time]["ann"] = ann_path

        paths_to_infos = lyft_helper.write_related_image_info(item._related_images, item.ann_data)
        time_to_data[time]["related_images"] = paths_to_infos
        return time_to_data

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        self._upload_dataset(api, dataset_id, log_progress, is_episodes=self._is_pcd_episode)

    def _upload_dataset(self, api: Api, dataset_id: int, log_progress=True, is_episodes=False):
        self._meta = ProjectMeta()
        meta, _, _ = self.merge_metas_with_conflicts(api, dataset_id)

        multiple_items = self.items_count > 1
        dataset_info = api.dataset.get_info_by_id(dataset_id)
        scene_name_to_dataset = {}

        if multiple_items:
            logger.info(f"Found {self.items_count} pointcloud files in the input data.")
            scene_names = set([item.scene_name for item in self._items])
            for name in scene_names:
                ds = api.dataset.create(
                    dataset_info.project_id,
                    name,
                    change_name_if_conflict=True,
                    parent_id=dataset_id,
                )
                scene_name_to_dataset[name] = ds

        if log_progress:
            progress, progress_cb = self.get_progress(self._total_msg_count, "Uploading...")
        else:
            progress_cb = None

        for idx, item in enumerate(self._items):
            current_dataset = scene_name_to_dataset.get(item.scene_name, None)
            if current_dataset is None:
                raise RuntimeError("Dataset not found for scene name: {}".format(item.scene_name))

            current_dataset_id = current_dataset.id
            time_to_data = self.convert(item, meta)
            if self.meta_needs_update:
                meta = api.project.update_meta(current_dataset.project_id, self._meta)
                self._meta = meta

            # existing_names = set([pcd.name for pcd in api.pointcloud.get_list(current_dataset_id)])
            ann_episode = PointcloudEpisodeAnnotation()
            frame_to_pointcloud_ids = {}

            for idx, (time, data) in enumerate(time_to_data.items()):
                pcd_path = data["pcd"]
                ann_path = data["ann"] or None
                pcd_meta = data.get("meta", {})  # todo ask about this
                related_images = data.get("related_images", {})

                if is_episodes:
                    pcd_meta["frame"] = idx
                upload_fn = (
                    api.pointcloud_episode.upload_path
                    if is_episodes
                    else api.pointcloud.upload_path
                )
                pcd_name = get_file_name(pcd_path)
                info = upload_fn(current_dataset_id, pcd_name, pcd_path, pcd_meta)
                pcd_id = info.id
                frame_to_pointcloud_ids[idx] = pcd_id

                if ann_path is not None:
                    ann = PointcloudAnnotation.load_json_file(ann_path, meta)
                    if is_episodes:
                        objects = ann_episode.objects
                        figures = []
                        for fig in ann.figures:
                            obj_cls = meta.get_obj_class(fig.parent_object.obj_class.name)
                            if obj_cls is not None:
                                obj = PointcloudEpisodeObject(obj_cls)
                                objects = objects.add(obj)
                                figure = PointcloudFigure(obj, fig.geometry, frame_index=idx)
                                figures.append(figure)
                        frames = ann_episode.frames
                        frames = frames.add(PointcloudEpisodeFrame(idx, figures))
                        ann_episode = ann_episode.clone(objects=objects, frames=frames)
                    else:
                        api.pointcloud.annotation.append(pcd_id, ann)

                rimage_infos = []
                camera_names = []
                for img_path, rimg_ann_path in related_images:
                    meta_json = load_json_file(rimg_ann_path)
                    img = api.pointcloud.upload_related_image(img_path)

                    camera_names.append(meta_json[ApiField.META]["deviceId"])
                    rimage_infos.append(
                        {
                            ApiField.ENTITY_ID: pcd_id,
                            ApiField.NAME: meta_json[ApiField.NAME],
                            ApiField.HASH: img,
                            ApiField.META: meta_json[ApiField.META],
                        }
                    )
                if len(rimage_infos) > 0:
                    api.pointcloud.add_related_images(rimage_infos, camera_names)

                silent_remove(pcd_path)
                if ann_path is not None:
                    silent_remove(ann_path)
                if related_images is not None:
                    for _, ann in related_images:
                        silent_remove(ann)
                if log_progress:
                    progress_cb(1)

            if is_episodes:
                ann_episode = ann_episode.clone(frames_count=len(time_to_data))
                api.pointcloud_episode.annotation.append(
                    current_dataset_id, ann_episode, frame_to_pointcloud_ids
                )

            logger.info(f"Dataset ID:{current_dataset_id} has been successfully updated.")

        if log_progress:
            if is_development():
                progress.close()
