import os

from supervisely import Annotation, ProjectMeta
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
            self._type = None
            self._shape = shape
            self._custom_data = custom_data
            self._meta = None

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

        def set_custom_data(self, custom_data: dict):
            self.custom_data = custom_data

        def update_custom_data(self, custom_data: dict):
            self.custom_data.update(custom_data)

        def update(self, item_path=None, ann_data=None, shape=None, custom_data={}):
            if item_path is not None:
                self.set_path(item_path)
            if ann_data is not None:
                self.set_ann_data(ann_data)
            if shape is not None:
                self.set_shape(shape)
            if custom_data:
                self.update_custom_data(custom_data)

    def __init__(self, data, items, annotations={}):
        self._input_data = data
        self._items = items
        self._annotations = annotations
        self._meta = None

    @property
    def format(self):
        return self.__str__()

    @property
    def items_count(self):
        return len(self._items)

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

        for path in self._annotations:
            is_valid = self.validate_ann_file(path)
            if not is_valid:
                return False
        # if self._meta is None:
        # return False
        return True

    def get_meta(self):
        if self._meta is not None:
            return self._meta
        else:
            return ProjectMeta()

    def get_items(self):  # -> generator?
        raise NotImplementedError()

    def to_supervisely(self, item: BaseItem, meta: ProjectMeta) -> Annotation:
        """Convert to Supervisely format."""
        if item.ann_data is None:
            if item.shape is not None:
                return Annotation(item.shape)
            else:
                return Annotation.from_img_path(item.path)
