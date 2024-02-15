from pycocotools.coco import COCO

from supervisely.convert.base_converter import AvailableImageFormats, BaseConverter

COCO_ANN_KEYS = ["images", "annotations", "categories"]

class COCOConverter(BaseConverter):
    def __init__(self, input_data):
        super().__init__(input_data)

    def __str__(self):
        return AvailableImageFormats.COCO

    @staticmethod
    def validate_ann_format(ann_path):
        coco = COCO(ann_path)  # dont throw error if not COCO
        if not all(key in coco.dataset for key in COCO_ANN_KEYS):
            return False
        return True
    
    def get_meta(self):
        return super().get_meta()
    
    def get_items(self):
        return super().get_items()

    def to_supervisely(self, image_path: str, ann_path: str):
        raise NotImplementedError()