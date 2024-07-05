import os
from typing import List, Tuple

import supervisely.convert.image.label_studio.label_studio_helper as helper
from supervisely import ProjectMeta, logger
from supervisely.annotation.annotation import Annotation
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import get_file_ext, get_file_name
from supervisely.io.json import load_json_file


class LabelStudioConverter(ImageConverter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source_ann_files = []
        self._source_images = {}

    def __str__(self):
        return AvailableImageConverters.LABEL_STUDIO

    @property
    def ann_ext(self) -> str:
        return ".json"

    def validate_key_file(self, ann_path: str) -> bool:
        """Validate LabelMe annotation file."""
        raw_ann = load_json_file(ann_path)
        if not isinstance(raw_ann, list):
            return False
        if len(raw_ann) == 0:
            return False
        if not all([isinstance(ann.get("data"), dict) for ann in raw_ann]):
            return False
        anns = []
        for ann in raw_ann:
            anns.extend(ann.get("annotations", []))
            anns.extend(ann.get("predictions", []))
        if not all([isinstance(ann, dict) for ann in anns]):
            return False
        if len(anns) == 0:
            return False
        if not any([ann.get("result") for ann in anns]):
            return False
        return True

    def validate_format(self) -> bool:
        logger.debug(f"Validating format: {self.__str__()}")
        anns_detected = self._find_images_with_annotations()
        if not anns_detected:
            logger.debug(f"No LabelStudio data found in {self._input_data}.")
            return False
        else:
            logger.debug(f"Found {self.items_count} LabelStudio items.")
            return True

    def _find_images_with_annotations(self) -> Tuple[List[ImageConverter.Item], ProjectMeta]:
        """Find images with annotations in Label Studio format."""

        for root, _, files in os.walk(self._input_data):
            for file in files:
                file_name_no_ext = get_file_name(file)
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if ext == self.ann_ext:
                    if self.validate_key_file(full_path):
                        self._source_ann_files.append(full_path)
                elif self.is_image(full_path):
                    self._source_images[file_name_no_ext] = full_path

        return len(self._source_ann_files) > 0

    def _create_items(self) -> None:
        """Create items from Label Studio annotations."""

        meta = ProjectMeta()
        items = []
        annotated_images = set()

        for ann_path in self._source_ann_files:
            raw_anns = load_json_file(ann_path)
            for raw_ann in raw_anns:
                annotations = raw_ann.get("annotations", [])
                annotations.extend(raw_ann.get("predictions", []))
                for ann in annotations:
                    image_path = raw_ann.get("data", {}).get("image")
                    if image_path is None:
                        continue
                    name_noext = get_file_name(image_path)
                    image_path = self._source_images.get(name_noext)
                    if image_path is None:
                        continue
                    sly_ann, meta = helper.create_supervisely_annotation(image_path, ann, meta)
                    item = self.Item(image_path, ann_data=sly_ann)
                    items.append(item)
                    if len(sly_ann.labels) > 0 or len(sly_ann.img_tags) > 0:
                        annotated_images.add(name_noext)

        no_ann_images = set(self._source_images.keys()) - annotated_images
        items.extend([self.Item(self._source_images[image_name]) for image_name in no_ann_images])

        self._items = items
        self._meta = meta

    def to_supervisely(
        self,
        item: ImageConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> Annotation:
        """Convert to Supervisely format."""

        item.set_shape()
        ann = item.ann_data
        if ann is None:
            return item.create_empty_annotation()
        if renamed_classes:
            new_labels = []
            for label in ann.labels:
                new_cls = renamed_classes.get(label.obj_class.name)
                if new_cls is not None:
                    obj_cls = meta.get_obj_class(new_cls)
                    label = label.clone(obj_class=obj_cls)
                new_labels.append(label)
            ann = ann.clone(labels=new_labels)
        return ann

    def upload_dataset(self, *args, **kwargs):
        """Upload dataset to Supervisely."""

        self._create_items()  # Create items before uploading

        return super().upload_dataset(*args, **kwargs)
