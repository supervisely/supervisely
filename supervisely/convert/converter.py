import imghdr
import os
from typing import Literal

from supervisely import Api, ProjectType, batched, logger
from supervisely.convert.base_converter import BaseConverter
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.video.video_converter import VideoConverter
from supervisely.imaging.image import is_valid_ext as is_valid_image_ext
from supervisely.io.fs import JUNK_FILES, get_file_ext
from supervisely.pointcloud.pointcloud import is_valid_ext as is_valid_point_cloud_ext
from supervisely.video.video import is_valid_ext as is_valid_video_ext
from supervisely.volume.volume import is_valid_ext as is_valid_volume_ext

possible_annotations_exts = [".json", ".xml", ".txt"]

# windows junk if endswith Zone.Identifier

# add class for Item with name, path, shape, ann data, ann path?


# class Converter:
class ImportManager:
    def __init__(self, input_data):
        # input_data date - folder / archive / link / team files
        # if save_path is None - save to the same level folder
        if not os.path.exists(input_data):
            raise RuntimeError(f"Directory does not exist: {input_data}")

        self._input_data = input_data
        self._items = []
        self._annotations = []

        self._modality = self._detect_modality()
        self._converter = self._get_converter()
        self._api = Api.from_env()

    @property
    def modality(self):
        return self._modality

    @property
    def converter(self):
        return self._converter

    def _detect_modality(self):
        """Detect modality of input data (images, videos, pointclouds, volumes)"""

        if self._contains_only_images():
            return ProjectType.IMAGES.value

        if self._contains_only_videos():
            return ProjectType.VIDEOS.value

        if self._contains_only_point_clouds():
            return ProjectType.POINT_CLOUDS.value  # @TODO: ProjectType.POINT_CLOUDS_EPISODES

        if self._contains_only_volumes():
            return ProjectType.VOLUMES.value
        else:
            raise RuntimeError("Use of mixed data types is not supported.")

    def _get_converter(self):
        """Return correct converter"""
        if self._modality == ProjectType.IMAGES.value:
            from supervisely.convert.image.coco.coco_converter import COCOConverter
            from supervisely.convert.image.pascal_voc.pascal_voc_converter import (
                PascalVOCConverter,
            )
            from supervisely.convert.image.sly.sly_image_converter import (
                SLYImageConverter,
            )
            from supervisely.convert.image.yolo.yolo_converter import YOLOConverter

            return ImageConverter(self._input_data, self._items, self._annotations)._converter
        elif self._modality == ProjectType.VIDEOS.value:
            return VideoConverter(self._input_data).converter
        # elif self.modality == ProjectType.POINT_CLOUDS.value:
        #     return PointCloudConverter(input_data)
        # elif self.modality == ProjectType.VOLUMES.value:
        #     return VolumeConverter(input_data)

    def upload_dataset(self, dataset_id):
        """Upload converted data to Supervisely"""
        meta = self.converter.get_meta()
        items = self.converter.get_items()
        for items_batch in batched(items, batch_size=50):

            item_names = [item.name for item in items_batch]
            item_paths = [item.path for item in items_batch]
            anns = []
            for item in items_batch:
                ann = self.converter.to_supervisely(item, meta)
                anns.append(ann)

            img_infos = self._api.image.upload_paths(dataset_id, item_names, item_paths)
            img_ids = [img_info.id for img_info in img_infos]
            self._api.annotation.upload_anns(img_ids, anns)

    def _contains_only_images(self):
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if file in JUNK_FILES:  # add better check
                    continue
                elif ext in possible_annotations_exts:
                    self._annotations.append(full_path)
                elif imghdr.what(full_path) is None:
                    logger.info(f"Non-image file found: {full_path}")
                    # self.items = [] # Clean items and annotations?
                    # self.annotations = []
                    return False
                else:
                    self._items.append(full_path)

        return True

    def _contains_only_videos(self):
        for root, _, files in os.walk(self._input_data):
            for file in files:
                ext = get_file_ext(file)
                if is_valid_video_ext(ext):
                    logger.info(f"Non-video file found: {os.path.join(root, file)}")
                    return False
        return True

    def _contains_only_point_clouds(self):
        for root, _, files in os.walk(self._input_data):
            for file in files:
                ext = get_file_ext(file)
                if is_valid_point_cloud_ext(ext):
                    logger.info(f"Non-point cloud file found: {os.path.join(root, file)}")
                    return False
        return True

    def _contains_only_volumes(self):
        for root, _, files in os.walk(self._input_data):
            for file in files:
                ext = get_file_ext(file)
                if is_valid_volume_ext(ext):
                    logger.info(f"Non-volume file found: {os.path.join(root, file)}")
                    return False
        return True
