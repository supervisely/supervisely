import imghdr
import os

from pycocotools.coco import COCO

import supervisely.convert.image.coco.coco_helper as coco_helper
from supervisely import Annotation, ProjectMeta
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import JUNK_FILES, get_file_ext

COCO_ANN_KEYS = ["images", "annotations", "categories"]


class COCOConverter(ImageConverter):
    def __init__(self, input_data):
        self._input_data = input_data
        self._items = []
        self._meta = None
        self._coco_categories = []

    def __str__(self) -> str:
        return AvailableImageConverters.COCO

    @property
    def ann_ext(self) -> str:
        return ".json"

    @property
    def key_file_ext(self) -> str:
        return ".json"

    def validate_ann_file(self, ann_data: dict, meta: ProjectMeta = None) -> bool:
        # TODO: implement detailed validation of COCO labels
        pass

    def generate_meta_from_annotation(self, coco: COCO, meta: ProjectMeta = None) -> ProjectMeta:
        return coco_helper.generate_meta_from_annotation(coco, meta)

    def validate_key_file(self, key_file_path) -> bool:
        coco = COCO(key_file_path)  # wont throw error if not COCO
        if not all(key in coco.dataset for key in COCO_ANN_KEYS):
            return False
        return True

    def validate_format(self) -> bool:
        detected_ann_cnt = 0
        images_list, ann_paths = [], []
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if ext == self.ann_ext:
                    is_valid = self.validate_key_file(full_path)
                    if is_valid:
                        ann_paths.append(full_path)
                        detected_ann_cnt += 1
                elif file in JUNK_FILES:
                    continue
                elif imghdr.what(full_path) is None:
                    # logger.info(f"Non-image file found: {full_path}")
                    return False
                else:
                    images_list.append(full_path)

        if len(ann_paths) is None:
            return False

        ann_dict = {}
        self._meta = ProjectMeta()
        for ann_path in ann_paths:
            coco = COCO(ann_path)
            coco_anns = coco.imgToAnns
            coco_images = coco.imgs
            coco_categories = coco.loadCats(ids=coco.getCatIds())
            self._coco_categories.extend(coco_categories)
            coco_items = coco_images.items()
            coco_meta = self.generate_meta_from_annotation(coco, self._meta)
            self._meta = self._meta.merge(coco_meta)
            # create ann dict
            for image_id, image_info in coco_items:
                image_name = image_info["file_name"]
                coco_ann = coco_anns[image_id]
                image_anns = ann_dict.get(image_name, None)
                if image_anns is None:
                    ann_dict[image_name] = coco_ann
                else:
                    ann_dict[image_name].extend(coco_ann)

        # create Items
        self._items = []
        for image_path in images_list:
            item = self.Item(image_path)
            if item.name in ann_dict:
                ann_data = ann_dict[item.name]
                # is_valid = self.validate_ann_file(ann_data, self._meta) in case of more detailed validation
                # if is_valid:
                item.set_ann_data(ann_data)
            self._items.append(item)
        return detected_ann_cnt > 0

    def get_meta(self) -> ProjectMeta:
        return self._meta

    def get_items(self) -> list:
        return self._items

    def to_supervisely(self, item: ImageConverter.Item, meta: ProjectMeta) -> Annotation:
        """Convert to Supervisely format."""
        if item.ann_data is None:
            return Annotation.from_img_path(item.path)
        else:
            return coco_helper.create_supervisely_annotation(item, meta, self._coco_categories)
