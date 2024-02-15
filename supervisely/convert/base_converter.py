from supervisely import Annotation
from supervisely.collection.str_enum import StrEnum
from supervisely.imaging.image import read


class AvailableImageFormats(StrEnum):
    SLY = "supervisely"
    COCO = "coco"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"

class AvailableVideoFormats(StrEnum):
    SLY = "supervisely"
    MOT = "coco"
    DAVIS = "yolo"

class BaseConverter:
    def __init__(self, data):
        self.input_data = data
        self.meta = None
        self.items = None# {"path/to/image.jpg": "path/to/annotation.json"}


    # def __str__(self):
    #     return "Base format converter."

    @property
    def format(self):
        return self.__str__()

    # @property
    # def dataset_name(self):
    #     pass

    # @property
    # def items_count(self):
    #     return len(self.items)

    # def _generate_project_meta(self):
    #     pass

    @staticmethod
    def validate_ann_format(ann_path):
        raise NotImplementedError()

    def get_meta(self):
        if self.meta is not None:
            return self.meta
        raise NotImplementedError()

    def get_items(self): # -> generator?
        raise NotImplementedError()
    
    def to_supervisely(self, item_path: str, ann_path: str) -> Annotation:
        """Convert to Supervisely format."""

        if self.meta is None:
            self.meta = self.get_meta()
        raise NotImplementedError()


    # def preview(self, sample_size=5):
    #     """Preview the sample data."""

    #     previews = []
    #     for i, (image_path, ann_path) in enumerate(self.get_items()):
    #         if i >= sample_size:
    #             break
    #         ann = self.to_supervisely(image_path, ann_path)
    #         img = read(image_path)
    #         ann.draw_pretty(img)
    #         previews.append(img)

    #     return previews
