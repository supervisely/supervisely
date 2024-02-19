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
    def __init__(self, input_data, project_type: ProjectType):
        if not os.path.exists(input_data):
            raise RuntimeError(f"Directory does not exist: {input_data}")

        self._input_data = input_data
        self._items = []
        self._modality = project_type
        self._converter = self.get_converter()
        self._api = Api.from_env()

    @property
    def modality(self):
        return self._modality

    @property
    def converter(self):
        return self._converter

    def get_converter(self):
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

            return ImageConverter(self._input_data)._converter
        elif self._modality == ProjectType.VIDEOS.value:
            return VideoConverter(self._input_data).converter
        # elif self.modality == ProjectType.POINT_CLOUDS.value:
        #     return PointCloudConverter(input_data)
        # elif self.modality == ProjectType.VOLUMES.value:
        #     return VolumeConverter(input_data)

    def upload_dataset(self, dataset_id):
        """Upload converted data to Supervisely"""
        self.converter.upload_dataset(self._api, dataset_id)
