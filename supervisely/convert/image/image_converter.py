import json
import os

from supervisely import logger
from supervisely.annotation.annotation import Annotation
from supervisely.convert.base_converter import BaseConverter

# from supervisely.convert.image.coco.coco_converter import COCOConverter
# from supervisely.convert.image.pascal_voc.pascal_voc_converter import PascalVOCConverter
# from supervisely.convert.image.sly.sly_image_converter import SLYImageConverter
# from supervisely.convert.image.yolo.yolo_converter import YOLOConverter
from supervisely.io.fs import get_file_ext, get_file_name_with_ext
from supervisely.io.json import load_json_file

ALLOWED_IMAGE_ANN_EXTENSIONS = [".json", ".txt", ".xml"]
# ALLOWED_CONVERTERS = [
#     COCOConverter,
#     PascalVOCConverter,
#     SLYImageConverter,
#     YOLOConverter,
# ]  # TODO: change


class ImageConverter(BaseConverter):
    class Item(BaseConverter.BaseItem):
        def __init__(self, item_path, ann_data=None, shape=None, custom_data={}):
            self._path = item_path
            self._ann_data = ann_data
            self._type = "image"
            self._shape = shape
            self._custom_data = custom_data
            # super().__init__(self.item_path, self.ann_data, self.shape)

    def __init__(self, input_data, items, annotations):
        self.input_data = input_data
        self.items = items
        self.annotations = annotations
        self.converter = self._detect_format()

    @property
    def format(self):
        return self.converter.format

    def _detect_format(self):
        found_formats = []

        all_converters = ImageConverter.__subclasses__()
        for converter in [all_converters[0]]:
            converter = converter(self.input_data, self.items, self.annotations)
            if converter.validate_format():
                if len(found_formats) > 1:
                    raise RuntimeError(
                        f"Multiple formats detected: {found_formats}. Mixed formats are not supported yet."
                    )
                found_formats.append(converter)

        if len(found_formats) == 0:
            logger.info(f"No valid dataset formats detected. Only image will be processed")
            return None

        if len(found_formats) == 1:
            return converter
