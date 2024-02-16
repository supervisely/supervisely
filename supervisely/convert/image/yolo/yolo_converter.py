import os

from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter


class YOLOConverter(ImageConverter):

    def __init__(self, input_data, items, annotations):
        self.input_data = input_data
        self.items = items
        self.annotations = annotations
        self.meta = None

    def __str__(self):
        return AvailableImageConverters.YOLO

    @staticmethod
    def validate_ann_format(ann_path):
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5 or not all(part.replace(".", "", 1).isdigit() for part in parts):
                    return False
        return True

    def require_key_file(self):
        return True

    def validate_key_file(self, key_path):
        return os.path.isfile(key_path)

    @property
    def get_ann_ext(self):  # ?
        return ".txt"

    def get_meta(self):
        raise NotImplementedError()

    def get_items(self):
        raise NotImplementedError()

    def to_supervisely(self):
        raise NotImplementedError()
