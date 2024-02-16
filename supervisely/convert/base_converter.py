import os

from supervisely import Annotation
from supervisely.collection.str_enum import StrEnum
from supervisely.imaging.image import read
from supervisely.io.fs import file_exists, get_file_ext, get_file_name_with_ext


class AvailableImageConverters:
    SLY = "supervisely"
    COCO = "coco"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"


class BaseConverter:
    class BaseItem:
        def __init__(self, item_path, ann_data=None, shape=None, custom_data={}):
            self._path = item_path
            self._ann_data = ann_data
            # self.ann_path = ann_path
            self._type = None
            self._shape = shape
            self._custom_data = custom_data

        @property
        def name(self):
            return get_file_name_with_ext(self._path)

        @property
        def path(self):
            return self._path

        @property
        def ann_data(self):
            return self._ann_data

        @property
        def type(self):
            return self._type

        @property
        def shape(self):
            return self._shape

        @property
        def custom_data(self):
            return self._custom_data

        def set_path(self, path):
            self._path = path

        def set_ann_data(self, ann_data):
            self._ann_data = ann_data

        def set_shape(self, shape):
            self._shape = shape

        def set_custom_data(self, custom_data):
            self.custom_data = custom_data

        def update_custom_data(self, custom_data):
            self.custom_data.update(custom_data)

    def __init__(self, data, items, annotations={}):
        self.input_data = data
        self.items = items  # {"path/to/image.jpg": "path/to/annotation.json"}
        self.annotations = annotations
        self.meta = None

    @property
    def format(self):
        return self.__str__()

    @property
    def items_count(self):
        return len(self.items)

    @property
    def ann_ext(self):
        raise NotImplementedError()

    @property
    def key_file_ext(self):
        raise NotImplementedError()

    @staticmethod
    def validate_ann_file(ann_path):
        raise NotImplementedError()

    def require_key_file(self):
        return False

    def validate_key_files(self):
        raise NotImplementedError()

    def validate_format(self):
        if self.require_key_file():
            self.validate_key_files()

        for path in self.annotations:
            is_valid = self.validate_ann_file(path)
            if not is_valid:
                return False
        if self.meta is None:
            return False
        return True

    def get_meta(self):
        if self.meta is not None:
            return self.meta
        raise NotImplementedError()

    def get_items(self):  # -> generator?
        raise NotImplementedError()

    def to_supervisely(self, item_path: str, ann_path: str, meta) -> Annotation:
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
