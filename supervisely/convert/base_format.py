from supervisely.collection.str_enum import StrEnum


class AvailableFormats(StrEnum):
    COCO = "coco"
    SLY = "supervisely"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"


class BaseFormat:
    def __init__(self, data):
        self.input_data = data
        self.items_mapping = {}  # {"path/to/image.jpg": "path/to/annotation.json"}

    # def __str__(self):
    #     return "Base format converter."

    @property
    def format(self):
        raise self.__str__()

    @property
    def dataset_name(self):
        raise self.__str__()

    @property
    def items_count(self):
        raise NotImplementedError()

    def generate_project_meta(self):
        pass

    def to_supervisely(self):
        raise NotImplementedError()

    def from_supervisely(self):
        raise NotImplementedError()

    def preview(self):
        """Preview the sample data."""
        raise NotImplementedError()
