from typing import Literal

from supervisely import ProjectType
from supervisely.convert.base_format import BaseFormat
from supervisely.convert.image.image_converter import ImageFormatConverter


class Converter:
    def __init__(
        self, input_data, save_path=None, output_type: Literal["folder", "archive"] = "folder"
    ):
        # input_data date - folder / archive / link / team files
        # if save_path is None - save to the same level folder
        self.input_data = input_data
        self.modality = self.detect_modality(input_data)

        if self.modality == ProjectType.IMAGES:
            return ImageFormatConverter(input_data)
        elif self.modality == ProjectType.VIDEOS:
            # return VideoFormatConverter(input_data)
            pass

    def detect_modality(self, data):
        """Detect modality of input data (images, videos, pointclouds, volumes)"""
        raise NotImplementedError()


# Using:
# converter: BaseFormat = Converter("/path/to/folder")  # read and return correct converter

# converter.to_supervisely()
# converter.to_yolo()
