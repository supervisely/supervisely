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
            logger.warn(
                'Install "lyft_dataset_sdk" python package to import datasets in LYFT format.'
            )
            return False

        def filter_fn(path):
            return all([(Path(path) / name).exists() for name in lyft_helper.FOLDER_NAMES])

        input_paths = [d for d in fs.dirs_filter(self._input_data, filter_fn)]
        if len(input_paths) == 0:
            return False
        input_path = input_paths[0]

        lidar_dir = input_path + "/lidar/"
        json_dir = input_path + "/data/"
        if lyft_helper.validate_ann_dir(json_dir) is False:
            return False

        bin_files = fs.list_files_recursively(
            lidar_dir, [self.key_file_ext], ignore_valid_extensions_case=True
        )

        if len(bin_files) == 0:
            return False

        # check if pointclouds have 5 columns (x, y, z, intensity, ring)
        pointcloud = np.fromfile(bin_files[0], dtype=np.float32)
        if pointcloud.shape[0] % 5 != 0:
            return False

        try:
            t = TinyTimer()
            lyft = Lyft(data_path=input_path, json_path=json_dir, verbose=False)
            self._lyft: Lyft = lyft
            logger.info(f"LyftDataset initialization took {t.get_sec():.2f} sec")
        except Exception as e:
            logger.info(f"Failed to initialize LyftDataset: {e}")
            return False

        t = TinyTimer()
        progress = Progress(f"Extracting annotations from available scenes...")
        # i = 0 # for debug
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
            progress.iter_done_report()
            # i += 1
            # if i == 2:
            #     break
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
        import open3d as o3d  # pylint: disable=import-error

        if getattr(item, "ann_data", None) is None:
            return PointcloudAnnotation()

        data = item.ann_data

        # * Convert annotation to json
        boxes = data["gt_boxes"]
        names = data["names"]

        objects = []
        for name, box in zip(names, boxes):
            center = [float(box[0]), float(box[1]), float(box[2])]
            size = [float(box[3]), float(box[5]), float(box[4])]
            ry = float(box[6])

            yaw = ry - np.pi
            yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi
            world_cam = None
            objects.append(o3d.ml.datasets.utils.BEVBox3D(center, size, yaw, name, -1.0, world_cam))
            objects[-1].yaw = ry

        geoms = [lyft_helper._convert_BEVBox3D_to_geometry(box) for box in objects]

        figures = []
        objs = []
        for l, geometry, token in zip(
            objects, geoms, data["instance_tokens"]
        ):  # by object in point cloud
            class_name = renamed_classes.get(l.label_class, l.label_class)
            tag_names = [
                self._lyft.get("attribute", attr_token).get("name", None)
                for attr_token in token["attribute_tokens"]
            ]
            tag_col = None
            if len(tag_names) > 0 and all([tag_name is not None for tag_name in tag_names]):
                tag_meta_names = [renamed_tags.get(name, name) for name in tag_names]
                tag_metas = [meta.get_tag_meta(tag_meta_name) for tag_meta_name in tag_meta_names]
                tag_col = PointcloudTagCollection([PointcloudTag(meta, None) for meta in tag_metas])
            pcobj = PointcloudObject(meta.get_obj_class(class_name), tag_col)
            figures.append(PointcloudFigure(pcobj, geometry))
            objs.append(pcobj)
        return PointcloudAnnotation(PointcloudObjectCollection(objs), figures)

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        unique_names = {name for item in self._items for name in item.ann_data["names"]}
        tag_names = {tag["name"] for tag in self._lyft.attribute}
        self._meta = ProjectMeta(
            [ObjClass(name, Cuboid3d) for name in unique_names],
            [TagMeta(tag, TagValueType.NONE) for tag in tag_names],
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

        for item in self._items:
            # * Get the current dataset for the scene
            current_dataset = scene_name_to_dataset.get(item._scene_name, None)
            if current_dataset is None:
                raise RuntimeError(f"Dataset not found for scene name: {item._scene_name}")
            current_dataset_id = current_dataset.id

            # * Convert timestamp to ISO format
            iso_time = datetime.utcfromtimestamp(item.ann_data["timestamp"] / 1e6).isoformat() + "Z"
            item.ann_data["timestamp"] = iso_time

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
