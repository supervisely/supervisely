import os
from pathlib import Path
from typing import Optional
from supervisely import PointcloudAnnotation, ProjectMeta, is_development, logger
from supervisely.api.api import Api, ApiField
from supervisely.convert.base_converter import AvailablePointcloudEpisodesConverters
from supervisely.convert.pointcloud_episodes.kitti_360 import kitti_360_helper
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import PointcloudEpisodeConverter
from supervisely.io.fs import (
    dirs_filter,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    list_files,
    list_files_recursively,
    silent_remove,
)
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)


class KITTI360Converter(PointcloudEpisodeConverter):

    class Item(PointcloudEpisodeConverter.Item):
        def __init__(
            self,
            frame_paths,
            labels,
            poses_path,
            related_images_paths: Optional[list] = None,
            custom_data: Optional[dict] = None,
        ):
            self._frame_paths = frame_paths
            self._labels = labels
            self._poses_path = poses_path
            self._related_images_paths = related_images_paths if related_images_paths is not None else []

            self._type = "point_cloud_episode"
            self._custom_data = custom_data if custom_data is not None else {}

    def __str__(self) -> str:
        return AvailablePointcloudEpisodesConverters.KITTI360

    @property
    def key_file_ext(self) -> str:
        return ".bin"

    def validate_format(self) -> bool:
        def _file_filter_fn(file_path):
            return get_file_ext(file_path).lower() == self.key_file_ext

        def _dir_filter_fn(path):
            return all([(Path(path) / name).exists() for name in kitti_360_helper.FOLDER_NAMES])

        input_paths = [d for d in dirs_filter(self._input_data, _dir_filter_fn)]
        if len(input_paths) == 0:
            return False

        input_path = input_paths[0]
        velodyne_dir = os.path.join(input_path, "data_3d_raw")
        poses_dir = os.path.join(input_path, "data_poses")
        boxes_dir = os.path.join(input_path, "data_3d_bboxes")
        calib_dir = os.path.join(input_path, "calibration")
        rimage_dir = os.path.join(input_path, "data_2d_raw")

        self._items = []
        velodyne_files = list_files_recursively(velodyne_dir, [self.key_file_ext], None, True)
        if len(velodyne_files) == 0:
            return False
        boxes_ann_files = list_files_recursively(boxes_dir, [".xml"], None, True)
        if len(boxes_ann_files) == 0:
            return False

        kitti_labels = []
        for ann_file in boxes_ann_files:
            key_name = Path(ann_file).stem
            frame_paths = []
            for path in velodyne_files:
                if key_name in Path(path).parts:
                    frame_paths.append(path)
            if len(frame_paths) == 0:
                logger.debug("No frames found for name: %s", key_name)
                continue
            rimage_path = os.path.join(rimage_dir, key_name, "image_00", "data_rect") # todo: check if this is correct
            rimage_paths = list_files(rimage_path, [".png"], None, True)

            poses_path = os.path.join(poses_dir, key_name, "cam0_to_world.txt")
            labels = kitti_360_helper.read_kitti_xml(ann_file, calib_dir)
            kitti_labels.extend(labels)
            self._items.append(self.Item(frame_paths, labels, poses_path, rimage_paths))

        self._meta = kitti_360_helper.convert_labels_to_meta(kitti_labels)
        return self.items_count > 0

    def to_supervisely(
        self,
        item: PointcloudEpisodeConverter.Item,
        meta: ProjectMeta,
        renamed_classes: dict = {},
        renamed_tags: dict = {},
    ) -> PointcloudAnnotation:
        label = item.ann_data
        objs, figures = kitti_360_helper.convert_label_to_annotation(label, meta, renamed_classes)
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
            kitti_360_helper.convert_bin_to_pcd(item.path, pcd_path)

            # * Upload pointcloud
            pcd_name = get_file_name_with_ext(pcd_path)
            info = api.pointcloud.upload_path(dataset_id, pcd_name, pcd_path, {})
            pcd_id = info.id

            # * Convert annotation and upload
            ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
            api.pointcloud.annotation.append(pcd_id, ann)

            # * Upload related images
            image_path, calib_path = item._related_images
            rimage_info = kitti_360_helper.convert_calib_to_image_meta(image_path, calib_path)

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
