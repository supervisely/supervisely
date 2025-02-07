import os
from collections import defaultdict
from typing import Dict, Optional, Union

import supervisely.convert.image.coco.coco_helper as coco_helper
from supervisely import Annotation, ProjectMeta
from supervisely.sly_logger import logger
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import JUNK_FILES, get_file_ext
from supervisely.project.project_settings import LabelingInterface


COCO_ANN_KEYS = ["images", "annotations"]


class COCOConverter(ImageConverter):
    def __init__(
            self,
            input_data: str,
            labeling_interface: Optional[Union[LabelingInterface, str]],
            upload_as_links: bool,
            remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)

        self._coco_categories = []
        self._supports_links = True
        self._force_shape_for_links = self.upload_as_links

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

    def generate_meta_from_annotation(self, coco, meta: ProjectMeta = None) -> ProjectMeta:
        return coco_helper.generate_meta_from_annotation(coco, meta)

    def validate_key_file(self, key_file_path) -> bool:
        from pycocotools.coco import COCO  # pylint: disable=import-error

        with coco_helper.HiddenCocoPrints():
            coco = COCO(key_file_path)  # wont throw error if not COCO
        if not all(key in coco.dataset for key in COCO_ANN_KEYS):
            return False
        return True

    def validate_format(self) -> bool:
        from pycocotools.coco import COCO  # pylint: disable=import-error

        if self.upload_as_links:
            self._download_remote_ann_files()
        detected_ann_cnt = 0
        images_list, ann_paths = [], []
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if ext == self.ann_ext:
                    ann_paths.append(full_path)
                elif file in JUNK_FILES:
                    continue
                elif self.is_image(full_path):
                    images_list.append(full_path)

        if len(ann_paths) == 0:
            return False

        ann_dict = {}
        meta = ProjectMeta()

        warnings = defaultdict(list)
        for ann_path in ann_paths:
            try:
                with coco_helper.HiddenCocoPrints():
                    coco = COCO(ann_path)
            except:
                continue
            if not all(key in coco.dataset for key in COCO_ANN_KEYS):
                continue
            coco_anns = coco.imgToAnns
            coco_images = coco.imgs
            if len(coco.cats) > 0:
                coco_categories = coco.loadCats(ids=coco.getCatIds())
            else:
                coco_categories = []
            self._coco_categories.extend(coco_categories)
            coco_items = coco_images.items()
            meta = self.generate_meta_from_annotation(coco, meta)
            # create ann dict
            for image_id, image_info in coco_items:
                image_name = image_info["file_name"]
                if not isinstance(image_name, str):
                    warnings["file_name field is not a string"].append(image_name)
                    continue

                if "/" in image_name:
                    image_name = os.path.basename(image_name)
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
                item.ann_data = ann_data
                detected_ann_cnt += 1
            self._items.append(item)

        self._meta = meta

        if len(warnings) > 0:
            for warning, failed_items in warnings.items():
                logger.warn(f"{warning}: {failed_items}")
        return detected_ann_cnt > 0

    def get_meta(self) -> ProjectMeta:
        return self._meta

    def get_items(self) -> list:
        return self._items

    def to_supervisely(
        self,
        item: ImageConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> Annotation:
        """Convert to Supervisely format."""
        if item.ann_data is None:
            return Annotation.from_img_path(item.path)
        else:
            if not self.upload_as_links:
                item.set_shape()
            ann = coco_helper.create_supervisely_annotation(
                item,
                meta,
                self._coco_categories,
                renamed_classes,
                renamed_tags,
            )
            return ann
