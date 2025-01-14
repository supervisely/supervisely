import os
from pathlib import Path

from supervisely import PointcloudAnnotation, ProjectMeta, is_development, logger
from supervisely.api.api import Api, ApiField
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.kitti_3d import kitti_3d_helper
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.io.fs import (
    dirs_filter,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    list_files,
    silent_remove,
)
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)


class KITTI3DConverter(PointcloudConverter):
    def __str__(self) -> str:
        return AvailablePointcloudConverters.KITTI3D

    @property
    def key_file_ext(self) -> str:
        return ".bin"

    def validate_format(self) -> bool:
        def _file_filter_fn(file_path):
            return get_file_ext(file_path).lower() == self.key_file_ext

        def _dir_filter_fn(path):
            return all([(Path(path) / name).exists() for name in kitti_3d_helper.FOLDER_NAMES])

        input_paths = [d for d in dirs_filter(self._input_data, _dir_filter_fn)]
        if len(input_paths) == 0:
            return False

        input_path = input_paths[0]
        velodyne_dir = os.path.join(input_path, "velodyne")
        image_2_dir = os.path.join(input_path, "image_2")
        label_2_dir = os.path.join(input_path, "label_2")
        calib_dir = os.path.join(input_path, "calib")

        self._items = []
        velodyne_files = list_files(velodyne_dir, filter_fn=_file_filter_fn)
        if len(velodyne_files) == 0:
            return False

        kitti_labels = []
        for velodyne_path in velodyne_files:
            file_name = get_file_name(velodyne_path)
            image_path = os.path.join(image_2_dir, f"{file_name}.png")
            label_path = os.path.join(label_2_dir, f"{file_name}.txt")
            calib_path = os.path.join(calib_dir, f"{file_name}.txt")
            if not file_exists(image_path):
                logger.debug(f"Skipping item: {velodyne_path}. Image not found.")
                continue
            if not file_exists(label_path):
                logger.debug(f"Skipping item: {velodyne_path}. Label not found.")
                continue
            if not file_exists(calib_path):
                logger.debug(f"Skipping item: {velodyne_path}. Calibration not found.")
                continue

            label = kitti_3d_helper.read_kitti_label(label_path, calib_path)
            kitti_labels.append(label)
            self._items.append(self.Item(velodyne_path, label, (image_path, calib_path)))

        self._meta = kitti_3d_helper.convert_labels_to_meta(kitti_labels)
        return self.items_count > 0

    def to_supervisely(
        self,
        item: PointcloudConverter.Item,
        meta: ProjectMeta,
        renamed_classes: dict = {},
        renamed_tags: dict = {},
    ) -> PointcloudAnnotation:
        label = item.ann_data
        objs, figures = kitti_3d_helper.convert_label_to_annotation(label, meta, renamed_classes)
        return PointcloudAnnotation(PointcloudObjectCollection(objs), figures)

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Converting pointclouds...")
        else:
            progress_cb = None

        for item in self._items:
            # * Convert pointcloud from ".bin" to ".pcd"
            pcd_path = str(Path(item.path).with_suffix(".pcd"))
            if file_exists(pcd_path):
                logger.warning(f"Overwriting file with path: {pcd_path}")
            kitti_3d_helper.convert_bin_to_pcd(item.path, pcd_path)

            # * Upload pointcloud
            pcd_name = get_file_name_with_ext(pcd_path)
            info = api.pointcloud.upload_path(dataset_id, pcd_name, pcd_path, {})
            pcd_id = info.id

            # * Convert annotation and upload
            ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
            api.pointcloud.annotation.append(pcd_id, ann)

            # * Upload related images
            image_path, calib_path = item._related_images
            rimage_info = kitti_3d_helper.convert_calib_to_image_meta(image_path, calib_path)

            image_jsons = []
            camera_names = []
            img = api.pointcloud.upload_related_image(image_path)
            image_jsons.append(
                {
                    ApiField.ENTITY_ID: pcd_id,
                    ApiField.NAME: get_file_name_with_ext(rimage_info[ApiField.NAME]),
                    ApiField.HASH: img,
                    ApiField.META: rimage_info[ApiField.META],
                }
            )
            camera_names.append(rimage_info[ApiField.META]["deviceId"])
            if len(image_jsons) > 0:
                api.pointcloud.add_related_images(image_jsons, camera_names)

            # * Clean up
            silent_remove(pcd_path)
            if log_progress:
                progress_cb(1)

        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")

        if log_progress:
            if is_development():
                progress.close()
