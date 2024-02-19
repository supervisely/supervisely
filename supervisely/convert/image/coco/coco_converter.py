import imghdr
import os

from pycocotools.coco import COCO

import supervisely.convert.image.coco.coco_helper as coco_helper
from supervisely import (
    Annotation,
    ObjClass,
    Polygon,
    ProjectMeta,
    Rectangle,
    TagMeta,
    TagValueType,
    logger,
)
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.imaging.color import generate_rgb
from supervisely.io.fs import JUNK_FILES, get_file_ext, list_files_recursively

COCO_ANN_KEYS = ["images", "annotations", "categories"]


class COCOConverter(ImageConverter):
    def __init__(self, input_data):
        self._input_data = input_data
        self._items = []
        self._meta = None

    def __str__(self):
        return AvailableImageConverters.COCO

    @property
    def ann_ext(self):
        return None  # ? ".json"

    @property
    def key_file_ext(self):
        return ".json"

    def validate_ann_file(self, ann_path: str):
        pass

    def validate_key_file(self, key_file_path):
        coco = COCO(key_file_path)  # wont throw error if not COCO
        if not all(key in coco.dataset for key in COCO_ANN_KEYS):
            return False
        return True

    def validate_format(self):
        detected_ann_cnt = 0

        # 1. find annotation file
        # 2. create image list and annotation dict {file_name: ann dict}
        # 3. create project meta from coco annotation
        # 4. implement to_supervisely() method

        images_list, ann_path = [], None
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if file in JUNK_FILES:
                    continue
                elif ext == self.ann_ext:
                    is_valid = self.validate_key_file(full_path)
                    if is_valid:
                        ann_path = full_path
                        detected_ann_cnt += 1
                elif imghdr.what(full_path) is None:
                    logger.info(f"Non-image file found: {full_path}")
                    return False
                else:
                    images_list.append(full_path)

        if ann_path is None:
            return False

        coco = COCO(ann_path)

        # create Items
        # @TODO: coco
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

    def get_items(self):  # -> generator?
        return self._items

    def to_supervisely(self, item: ImageConverter.Item, meta: ProjectMeta) -> Annotation:
        """Convert to Supervisely format."""
        if item.ann_data is None:
            if item.shape is not None:
                return Annotation(item.shape)
            else:
                return Annotation.from_img_path(item.path)

        ann = coco_helper.create_supervisely_ann(
            meta, item.custom_data["categories"], item.ann_data, item.shape
        )
        return ann

    def validate_format(self):
        return False
