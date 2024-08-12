import os
from collections import defaultdict
from typing import Dict, Generator, Union

import cv2
import numpy as np

from supervisely import ProjectMeta, generate_free_name, is_development, logger
from supervisely.api.api import Api, ApiContext
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import get_file_ext, list_files_recursively
from supervisely.project.project_settings import LabelingInterface

# def convert_to_nrrd(image, save_path):
#     """Convert image to nrrd format"""
#     image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#     image = sly.image.fliplr(image)
#     nrrd.write(save_path, image)


class HighColorDepthImageConverter(ImageConverter):
    allowed_exts = [".png", ".tiff", ".tif", ".bmp", ".exr", ".hdr"]

    def __str__(self):
        return AvailableImageConverters.HIGH_COLOR_DEPTH

    def validate_labeling_interface(self) -> bool:
        """Only multi_view labeling interface can be used to group images on single screen."""
        return self._labeling_interface in [
            LabelingInterface.DEFAULT,
            LabelingInterface.IMAGES_WITH_16_COLOR,
        ]

    def validate_format(self) -> bool:
        logger.debug(f"Validating format: {self.__str__()}")

        for image_path in self._find_high_color_depth_image_generator(self._input_data):
            item = self.Item(image_path)
            self._items.append(item)

        if self.items_count == 0:
            logger.debug(f"Not found any images with high color depth in '{self._input_data}'")
            return False

        return True

    def _find_high_color_depth_image_generator(self, input_data: str) -> Generator[str, None, None]:
        for image_path in list_files_recursively(input_data, valid_extensions=self.allowed_exts):
            if self._is_high_color_depth_image(image_path):
                yield image_path

    def _is_high_color_depth_image(self, image_path: str) -> bool:
        ext = get_file_ext(image_path).lower()
        if ext in [".exr", ".hdr"]:
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img.dtype in self._supported_depths:
                return True
        elif ext in [".tiff", ".tif"]:
            import tifffile

            img = tifffile.imread(image_path)
            if img.dtype in self._supported_depths:
                return True

        return True

    @property
    def _supported_depths(self):
        return [np.uint16, np.int16, np.float16, np.float32, np.float64]

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
    ) -> None:
        """Upload converted data to Supervisely"""
        super().upload_dataset(api, dataset_id, batch_size, log_progress)
