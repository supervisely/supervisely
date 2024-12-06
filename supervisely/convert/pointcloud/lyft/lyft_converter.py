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
    LabelingInterface,
    silent_remove,
)
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely.io.fs import get_file_ext, get_file_name, list_files_recursively, silent_remove
from . import lyft_helper


class LyftPointcloudConverter(PointcloudConverter):
    class Item(PointcloudConverter.Item):
        def __init__(
            self,
            item_path,
            ann_data: str = None,
            related_images: list = None,
            custom_data: dict = None,
        ):
            super().__init__(item_path, ann_data, related_images, custom_data)
            self._type = "point_cloud"

    def __init__(
        self,
        input_data: str,
        labeling_interface: Optional[Union[LabelingInterface, str]],
        upload_as_links: bool,
        remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)
        self._total_msg_count = 0
        self._is_pcd_episode = False

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
            if pointcloud % 5 != 0:
                return False
        
        json_path = self._input_data # todo: find the json folder
        lyft = Lyft(data_path=self._input_data, json_path=json_path, verbose=False)
        for scene in lyft_helper.get_available_scenes(lyft):
            sample_datas = lyft_helper.extract_data_from_scene(lyft, scene)
            if sample_datas is None:
                return
            for sample_data in sample_datas:
                item_path = sample_data['lidar_path']
                ann_data = sample_data['ann_data']
                related_images = [img_path for sensor, img_path in sample_data['ann_data'].items() if "CAM" in sensor]
                custom_data = sample_data.get("custom_data", {}) # todo: implement
                item = self.Item(item_path, ann_data, related_images, custom_data)
                self._items.append(item)

        return self.items_count > 0

    def convert(self, item: PointcloudConverter.Item, meta: ProjectMeta):
        item_path = item.item_path
        save_path = item_path[:-4] + ".pcd"
        timestamp = os.path.getmtime(item.item_path)
        time_to_data = lyft_helper.convert_bin_to_pcd(item_path, save_path, timestamp)
        raise NotImplementedError("Implement the conversion function")
        return time_to_data

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        self._upload_dataset(api, dataset_id, log_progress, is_episodes=self._is_pcd_episode)

    def _upload_dataset(self, api: Api, dataset_id: int, log_progress=True, is_episodes=False):
        obj_cls = ObjClass("object", Cuboid3d)
        self._meta = ProjectMeta(obj_classes=[obj_cls])
        meta, _, _ = self.merge_metas_with_conflicts(api, dataset_id)

        multiple_items = self.items_count > 1
        datasets = []
        dataset_info = api.dataset.get_info_by_id(dataset_id)

        if multiple_items:
            logger.info(
                f"Found {self.items_count} pointcloud files in the input data."
                "Will create dataset in parent dataset for each file."
            )
            nested_datasets = api.dataset.get_list(dataset_info.project_id, parent_id=dataset_id)
            existing_ds_names = set([ds.name for ds in nested_datasets])

            for item in self._items:
                ds_name = generate_free_name(existing_ds_names, get_file_name(item.path))
                ds = api.dataset.create(
                    dataset_info.project_id,
                    ds_name,
                    change_name_if_conflict=True,
                    parent_id=dataset_id,
                )
                existing_ds_names.add(ds.name)
                datasets.append(ds)

        if log_progress:
            progress, progress_cb = self.get_progress(self._total_msg_count, "Uploading...")
        else:
            progress_cb = None

        for idx, item in enumerate(self._items):
            current_dataset = dataset_info if not multiple_items else datasets[idx]
            current_dataset_id = current_dataset.id
            time_to_data = self.convert(item, meta)

            existing_names = set([pcd.name for pcd in api.pointcloud.get_list(current_dataset_id)])
            ann_episode = PointcloudEpisodeAnnotation()
            frame_to_pointcloud_ids = {}

            for idx, (time, data) in enumerate(time_to_data.items()):
                pcd_path = data["pcd"].as_posix()
                ann_path = data["ann"].as_posix() if data["ann"] is not None else None
                pcd_meta = data["meta"]

                pcd_name = generate_free_name(
                    existing_names, f"{time}.pcd", with_ext=True, extend_used_names=True
                )
                if is_episodes:
                    pcd_meta["frame"] = idx
                upload_fn = (
                    api.pointcloud_episode.upload_path
                    if is_episodes
                    else api.pointcloud.upload_path
                )
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

                silent_remove(pcd_path)
                if ann_path is not None:
                    silent_remove(ann_path)
                if log_progress:
                    progress_cb(1)

            if is_episodes:
                ann_episode = ann_episode.clone(frames_count=len(time_to_data))
                api.pointcloud_episode.annotation.append(
                    current_dataset_id, ann_episode, frame_to_pointcloud_ids
                )

            logger.info(f"Dataset ID:{current_dataset_id} has been successfully uploaded.")

        if log_progress:
            if is_development():
                progress.close()
