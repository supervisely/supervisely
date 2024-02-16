import xml.etree.ElementTree as ET

from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter


class PascalVOCConverter(ImageConverter):

    def __init__(self, input_data, items, annotations):
        self.input_data = input_data
        self.items = items
        self.annotations = annotations
        self.meta = None

    def __str__(self):
        return AvailableImageConverters.PASCAL_VOC

    @staticmethod
    def validate_ann_format(ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        if root.tag == "annotation":
            return "Pascal VOC"

    def get_meta(self):
        return super().get_meta()

    def get_items(self):
        return super().get_items()

    def to_supervisely(self, image_path: str, ann_path: str):
        raise NotImplementedError()
