from supervisely.convert.base_converter import AvailableImageFormats, BaseConverter
from supervisely.io.json import load_json_file

SLY_ANN_KEYS = ["imageName", "imageId", "createdAt", "updatedAt", "annotation"]

class SLYImageConverter(BaseConverter):
    def __init__(self, input_data):
        super().__init__(input_data)

    def __str__(self):
        return AvailableImageFormats.SLY
    
    @staticmethod
    def validate_ann_format(ann_path):
        ann_json = load_json_file(ann_path)
        if all(key in ann_json for key in SLY_ANN_KEYS):
            return True
        return False
                
    def get_meta(self):
        return super().get_meta()
    
    def get_items(self):
        return super().get_items()

    def to_supervisely(self):
        raise NotImplementedError()

    def to_coco(self):
        raise NotImplementedError()

    def to_pascal_voc(self):
        raise NotImplementedError()

    def to_yolo(self):
        raise NotImplementedError()
