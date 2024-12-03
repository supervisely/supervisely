import os
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
from collections import defaultdict

from supervisely import (
    Api,
    ObjClass,
    PointcloudAnnotation,
    ProjectMeta,
    generate_free_name,
    logger,
    is_development,
    LabelingInterface,
)
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.io.fs import get_file_ext, get_file_name, list_files_recursively, silent_remove
from .lyft_helper import process_pointcloud_msg


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

    def __str__(self) -> str:
        return AvailablePointcloudConverters.LYFT

    @property
    def key_file_ext(self) -> str:
        return ".bin"

    def validate_format(self) -> bool:
        def _filter_fn(file_path):
            return get_file_ext(file_path).lower() == self.key_file_ext

        lyft_files = list_files_recursively(self._input_data, filter_fn=_filter_fn)

        if len(lyft_files) == 0:
            return False

        self._items = []
        for file_path in lyft_files:
            item = self.Item(file_path)
            self._items.append(item)
        return self.items_count > 0

    def convert(self, item: PointcloudConverter.Item, meta: ProjectMeta):
        pointcloud = np.fromfile(item.item_path, dtype=np.float32).reshape(-1, 5)
        timestamp = os.path.getmtime(item.item_path)
        time_to_data = defaultdict(dict)

        process_pointcloud_msg(
            time_to_data, pointcloud, timestamp, Path(item.item_path), meta, is_ann=False
        )

        return time_to_data

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        self._upload_dataset(api, dataset_id, log_progress)

    def _upload_dataset(self, api: Api, dataset_id: int, log_progress=True):
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

        existing_names = set([pcd.name for pcd in api.pointcloud.get_list(current_dataset_id)])
        for idx, item in enumerate(self._items):
            current_dataset = dataset_info if not multiple_items else datasets[idx]
            current_dataset_id = current_dataset.id

            time_to_data = self.convert(item, meta)
            for time, data in time_to_data.items():
                pcd_path = data["pcd"].as_posix()
                ann_path = data["ann"].as_posix() if data["ann"] is not None else None
                pcd_meta = data["meta"]

                pcd_name = generate_free_name(
                    existing_names, f"{time}.pcd", with_ext=True, extend_used_names=True
                )
                info = api.pointcloud.upload_path(current_dataset_id, pcd_name, pcd_path, pcd_meta)
                pcd_id = info.id

                ann = PointcloudAnnotation()
                api.pointcloud.annotation.append(pcd_id, ann)

                silent_remove(pcd_path)
                if ann_path is not None:
                    silent_remove(ann_path)
                if log_progress:
                    progress_cb(1)

        if log_progress:
            if is_development():
                progress.close()

        logger.info(f"Dataset ID:{current_dataset_id} has been successfully uploaded.")
