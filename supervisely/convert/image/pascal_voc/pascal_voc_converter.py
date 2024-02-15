import xml.etree.ElementTree as ET

from supervisely.convert.base_converter import AvailableImageFormats, BaseConverter


class PascalVOCConverter(BaseConverter):
    def __init__(self, input_data):
        super().__init__(input_data)

    def __str__(self):
        return AvailableImageFormats.PASCAL_VOC
    
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
