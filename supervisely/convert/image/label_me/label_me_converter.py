import os
from typing import List, Tuple

import supervisely.convert.image.label_me.label_me_helper as label_me_helper
from supervisely import ProjectMeta, logger
from supervisely.annotation.annotation import Annotation
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import get_file_ext, get_file_name
from supervisely.io.json import load_json_file


class LabelmeConverter(ImageConverter):

    def __str__(self):
        return AvailableImageConverters.LABEL_ME

    @property
    def ann_ext(self) -> str:
        return ".json"

    def validate_ann_file(self, ann_path: str) -> bool:
        """Validate LabelMe annotation file."""

        ann_json = load_json_file(ann_path)
        if not isinstance(ann_json, dict):
            return False
        shapes = ann_json.get("shapes")
        if shapes is None or not isinstance(shapes, list):
            return False
        if ann_json.get("imagePath") is None and ann_json.get("imageData") is None:
            return False
        return True

    def validate_format(self) -> bool:
        logger.debug(f"Validating format: {self.__str__()}")
        items, meta = self._find_images_with_annotations()
        if len(items) == 0:
            logger.debug(f"No LabelMe data found in {self._input_data}.")
            return False
        else:
            self._items = items
            self._meta = meta
            logger.debug(f"Found {self.items_count} LabelMe items.")
            return True

    def _find_images_with_annotations(self) -> Tuple[List[ImageConverter.Item], ProjectMeta]:
        items = []
        images, anns = {}, {}
        meta = ProjectMeta()
        for root, _, files in os.walk(self._input_data):
            for file in files:
                file_name_no_ext = get_file_name(file)
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if ext == self.ann_ext:
                    if self.validate_ann_file(full_path):
                        meta = label_me_helper.update_meta_from_labelme_annotation(meta, full_path)
                        anns[file_name_no_ext] = full_path
                elif self.is_image(full_path):
                    images[file_name_no_ext] = full_path
        for name_noext, ann_path in anns.items():
            image_path = label_me_helper.get_image_from_data(ann_path, images.get(name_noext))
            if image_path is not None:
                item = self.Item(image_path, ann_path)
                items.append(item)
        return items, meta

    def to_supervisely(
        self,
        item: ImageConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> Annotation:
        """Convert to Supervisely format."""

        item.set_shape()
        ann = label_me_helper.create_supervisely_annotation(
            item,
            meta,
            renamed_classes,
        )
        return ann
