import imghdr
import os

import supervisely.convert.image.sly.sly_image_helper as sly_image_helper
from supervisely import Annotation, ProjectMeta, logger
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import JUNK_FILES, get_file_ext

from supervisely.io.json import load_json_file

# SLY_ANN_KEYS = ["imageName", "imageId", "createdAt", "updatedAt", "annotation"]


class SLYImageConverter(ImageConverter):
    def __init__(self, input_data):
        self._input_data = input_data
        self._items = []
        self._meta = None

    def __str__(self):
        return AvailableImageConverters.SLY

    @property
    def ann_ext(self):
        return ".json"

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta):
        try:
            ann_json = load_json_file(ann_path)["annotation"]
            ann = Annotation.from_json(ann_json, meta)
            return True
        except:
            return False

    def validate_key_file(self, key_file_path: str):
        try:
            self._meta = ProjectMeta.from_json(load_json_file(key_file_path))
            return True
        except Exception:
            return False

    def validate_format(self):
        detected_ann_cnt = 0
        meta_path = None
        images_list, ann_dict = [], {}
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                if file == "meta.json":
                    is_valid = self.validate_key_file(full_path)
                    if is_valid:
                        continue

                ext = get_file_ext(full_path)
                if file in JUNK_FILES:  # add better check
                    continue
                elif ext in self.ann_ext:
                    ann_dict[file] = full_path
                elif imghdr.what(full_path) is None:
                    logger.info(f"Non-image file found: {full_path}")
                    return False
                else:
                    images_list.append(full_path)

        if self._meta is not None:
            meta = self._meta
        else:
            meta = ProjectMeta()

        # create Items
        self._items = []
        for image_path in images_list:
            item = self.Item(image_path)
            ann_name = f"{item.name}.json"
            if ann_name in ann_dict:
                ann_path = ann_dict[ann_name]
                if self._meta is None:
                    meta = self.generate_meta_from_annotation(ann_path, meta)
                is_valid = self.validate_ann_file(ann_path, meta)
                if is_valid:
                    item.set_ann_data(ann_path)
                    detected_ann_cnt += 1
            self._items.append(item)
        self._meta = meta
        return detected_ann_cnt > 0

    def get_meta(self):
        return self._meta

    def generate_meta_from_annotation(self, ann_path, meta):
        meta = sly_image_helper.get_meta_from_annotation(meta, ann_path)
        return meta

    def get_items(self):
        return self._items

    def to_supervisely(self, item: ImageConverter.Item, meta: ProjectMeta = None) -> Annotation:
        """Convert to Supervisely format."""
        if meta is None:
            meta = self._meta

        if item.ann_data is None:
            return item.create_empty_annotation()

        try:
            ann_json = load_json_file(item.ann_data)["annotation"]
            return Annotation.from_json(ann_json, meta)
        except Exception as e:
            logger.warn(f"Failed to convert annotation: {repr(e)}")
            return item.create_empty_annotation()
