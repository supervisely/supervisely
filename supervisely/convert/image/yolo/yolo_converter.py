import os

from supervisely.convert.base_converter import AvailableImageFormats, BaseConverter


class YOLOConverter(BaseConverter):
    def __init__(self, input_data):
        super().__init__(input_data)
        
    def __str__(self):
        return AvailableImageFormats.YOLO
    
    @staticmethod
    def validate_ann_format(ann_path):
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5 or not all(
                    part.replace(".", "", 1).isdigit() for part in parts
                ):
                    return False
        return True

    def require_key_file(self):
        return True

    def validate_key_file(self, key_path):
        return os.path.isfile(key_path)
    @property
    def get_ann_ext(self): # ?
        return ".txt"

    def get_meta(self):
        raise NotImplementedError()
    
    def get_items(self):
        raise NotImplementedError()
    
    def to_supervisely(self):
        raise NotImplementedError()
