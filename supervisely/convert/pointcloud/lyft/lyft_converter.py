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
)
from supervisely.io import fs
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.convert.pointcloud.lyft import lyft_helper
from supervisely.api.api import ApiField
from datetime import datetime
from supervisely import TinyTimer


class LyftConverter(PointcloudConverter):
    class Item(PointcloudConverter.Item):

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
        self._is_pcd_episode = False
        self._lyft = None

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

        def filter_fn(path):
            return Path(path).name in ["data", "lidar", "images", "maps"]

        dirs = [lyft_dir for lyft_dir in fs.dirs_filter(self._input_data, filter_fn)]
        if len(dirs) != 4:
            return False

        lidar_dir = self._input_data + "/lidar/"
        json_dir = self._input_data + "/data/"

        bin_files = fs.list_files_recursively(
            lidar_dir, [self.key_file_ext], ignore_valid_extensions_case=True
        )

        if len(bin_files) == 0:
            return False

        # check if pointclouds have 5 columns (x, y, z, intensity, ring)
        pointcloud = np.fromfile(bin_files[0], dtype=np.float32)
        if pointcloud.shape[0] % 5 != 0:
            return False

        t = TinyTimer()
        lyft = Lyft(data_path=self._input_data, json_path=json_dir, verbose=False)
        self._lyft = lyft
        logger.info(f"LyftDataset initialization took {t.get_sec():.2f} sec")

        t = TinyTimer()
        progress = Progress(f"Extracting annotations from available scenes...")
        for scene in lyft_helper.get_available_scenes(lyft):
            scene_name = scene["name"]
            sample_datas = lyft_helper.extract_data_from_scene(lyft, scene)
            if sample_datas is None:
                logger.warning(f"Failed to extract sample data from scene: {scene['name']}.")
                continue
            for sample_data in sample_datas:
                item_path = sample_data["lidar_path"]
                ann_data = sample_data["ann_data"]
                related_images = lyft_helper.get_related_images(ann_data)
                custom_data = sample_data.get("custom_data", {})
                item = self.Item(item_path, ann_data, related_images, custom_data, scene_name)
                self._items.append(item)
            # self._scene_to_sample_cnt[scene_name] = len(sample_datas)
            progress.iter_done_report()
            break  # ! remove
        t = t.get_sec()
        logger.info(
            f"Lyft annotation extraction took {t:.2f} sec ({(t / self.items_count):.3f} sec per sample)"
        )

        return self.items_count > 0

    def to_supervisely(
        self,
        item: PointcloudConverter.Item,
        meta: ProjectMeta,
        renamed_classes: dict = {},
        renamed_tags: dict = {},
    ) -> PointcloudAnnotation:
        """
        Converts a point cloud item and its annotations to the supervisely formats.

        Args:
            item (PointcloudConverter.Item): The point cloud item to convert.
            meta (ProjectMeta): The project meta.

        Returns:
            PointcloudAnnotation: The converted point cloud annotation.
        """
        if getattr(item, "ann_data", None) is None:
            return None

        # * Convert annotation to json
        label = lyft_helper.lyft_annotation_to_BEVBox3D(item.ann_data)

        return lyft_helper.convert_label_to_annotation(label, meta, renamed_classes)

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        unique_names = {name for item in self._items for name in item.ann_data["names"]}
        self._meta = ProjectMeta([ObjClass(name, Cuboid3d) for name in unique_names])
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

        for item in self._items:
            # * Get the current dataset for the scene
            current_dataset = scene_name_to_dataset.get(item._scene_name, None)
            if current_dataset is None:
                raise RuntimeError(f"Dataset not found for scene name: {item._scene_name}")
            current_dataset_id = current_dataset.id

            # * Convert pointcloud from ".bin" to ".pcd"
            pcd_path = str(Path(item.path).with_suffix(".pcd"))
            if fs.file_exists(pcd_path):
                logger.warning(f"Overwriting file with path: {pcd_path}")
            lyft_helper.convert_bin_to_pcd(item.path, pcd_path)

            # * Upload pointcloud
            pcd_name = fs.get_file_name(pcd_path)
            info = api.pointcloud.upload_path(current_dataset_id, pcd_name, pcd_path, {})
            pcd_id = info.id

            # * Convert annotation and upload
            ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
            api.pointcloud.annotation.append(pcd_id, ann)

            # * Upload related images
            image_jsons = []
            camera_names = []
            for img_path, rimage_info in lyft_helper.generate_rimage_infos(
                item._related_images, item.ann_data
            ):
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

            # * Clean up
            fs.silent_remove(pcd_path)
            if log_progress:
                progress_cb(1)

        logger.info(f"Dataset ID:{current_dataset_id} has been successfully uploaded.")

        if log_progress:
            if is_development():
                progress.close()
