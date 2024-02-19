import os

import supervisely.convert.image.sly.sly_image_helper as sly_image_helper
from supervisely import Annotation, ProjectMeta
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.geometry.geometry import Geometry
from supervisely.io.fs import file_exists, get_file_ext, list_files_recursively
from supervisely.io.json import load_json_file

SLY_ANN_KEYS = ["imageName", "imageId", "createdAt", "updatedAt", "annotation"]


# match items and anns on init?


class SLYImageConverter(ImageConverter):
    def __init__(self, input_data, items, annotations):
        self._input_data = input_data
        self._items = items
        self._annotations = annotations
        self._meta = None

    def __str__(self):
        return AvailableImageConverters.SLY

    @property
    def ann_ext(self):
        return ".json"

    def validate_ann_file(self, ann_path):
        if self._meta is None:
            if file_exists(ann_path):
                ann_json = load_json_file(ann_path)
                if all(key in ann_json for key in SLY_ANN_KEYS):
                    return True
            return False
        else:
            try:
                ann = Annotation.from_json(load_json_file(ann_path), self._meta)
                return True
            except Exception:
                return False

    def require_key_file(self):
        return True

    def validate_key_files(self):
        jsons = list_files_recursively(self._input_data, valid_extensions=[".json"])
        # TODO: find meta.json first
        for key_file in jsons:
            try:
                self._meta = ProjectMeta.from_json(load_json_file(key_file))
                return True
            except Exception:
                continue
        return False

    def get_meta(self):
        if self._meta is not None:
            return self._meta
        else:
            return self.generate_meta_from_annotations()

    def generate_meta_from_annotations(self):
        meta = sly_image_helper.get_meta_from_annotations(self._annotations, self.validate_ann_file)
        return meta

    def get_items(self):
        return self._items

    def to_supervisely(self, item: ImageConverter.Item, meta: ProjectMeta) -> Annotation:
        """Convert to Supervisely format."""
        if self._meta is None:
            self._meta = self.get_meta()

        if item.ann_data is None:
            return item.create_empty_annotation()

        if isinstance(item.ann_data, dict):
            return Annotation.from_json(item.ann_data, meta)
        elif isinstance(item.ann_data, str):
            if file_exists(item.ann_data):
                return Annotation.from_json(load_json_file(item.ann_data), meta)
        elif isinstance(item.ann_data, Annotation):  # not intended usecase
            return item.ann_data
        else:
            return item.create_empty_annotation()  #  or raise?
