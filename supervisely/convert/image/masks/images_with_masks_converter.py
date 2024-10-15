import os
from typing import Dict, Optional, Union

import supervisely.convert.image.masks.image_with_masks_helper as helper
from supervisely import (
    Annotation,
    ProjectMeta,
    logger,
    ObjClass,
    Bitmap,
    ObjClassCollection,
    Rectangle,
)
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import file_exists, dirs_with_marker, dir_exists, get_file_name, list_files, dirs_filter, remove_junk_from_dir, get_file_ext
from supervisely.io.json import load_json_file
from supervisely.project.project_settings import LabelingInterface
from supervisely.convert.image.image_helper import validate_image_bounds

class ImagesWithMasksConverter(ImageConverter):
    def __init__(
            self,
            input_data: str,
            labeling_interface: Optional[Union[LabelingInterface, str]],
            upload_as_links: bool,
            remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)

        self._classes_mapping = {}

    def __str__(self):
        return AvailableImageConverters.MASKS

    @property
    def key_file_ext(self) -> str:
        return ".json"

    def validate_key_file(self, key_file_path: str) -> bool:
        if not file_exists(key_file_path):
            return False
        try:
            classes_mapping = load_json_file(key_file_path)
            if not isinstance(classes_mapping, dict):
                return False
            cls_list = [ObjClass(cls_name, Bitmap) for cls_name in classes_mapping]
            obj_class_collection = ObjClassCollection(cls_list)
            self._classes_mapping = classes_mapping
            return ProjectMeta(obj_classes=obj_class_collection)
        except Exception as e:
            logger.warn(f"Failed to read obj_class_to_machine_color.json: {repr(e)}")
            return False

    def validate_format(self) -> bool:
        detected_ann_cnt = 0
        possible_dirs = [d for d in dirs_with_marker(self._input_data, helper.COLOR_MAP_FILE_NAME)]
        if len(possible_dirs) == 0:
            return False
        if len(possible_dirs) > 1:
            return False
        project_dir = possible_dirs[0]
        project_dir_name = os.path.basename(project_dir.rstrip("/"))
        colors_file = os.path.join(project_dir, helper.COLOR_MAP_FILE_NAME)
        key_file_result = self.validate_key_file(colors_file)
        if key_file_result is False:
            return False
        self._meta = key_file_result

        # possible_dss
        def _search_for_dss(dir_path):
            if any([d in os.listdir(dir_path) for d in helper.MASK_DIRS]):
                return True
            return False

        dataset_paths = [d for d in dirs_filter(project_dir, _search_for_dss)]
        dataset_names = [os.path.basename(d.rstrip("/")) for d in dataset_paths]

        self._items = []
        for dataset_path, dataset_name in zip(dataset_paths, dataset_names):
            img_dir = os.path.join(dataset_path, helper.IMAGE_DIR_NAME)
            if not dir_exists(img_dir):
                continue
            names = [get_file_name(file_name) for file_name in os.listdir(img_dir)]
            img_paths = [os.path.join(img_dir, file_name) for file_name in os.listdir(img_dir)]

            machine_masks_dir = os.path.join(dataset_path, helper.MASKS_MACHINE_DIR_NAME)
            instance_masks_dir = os.path.join(dataset_path, helper.MASKS_INSTANCE_DIR_NAME)
            machine_masks_dir_exists = dir_exists(machine_masks_dir)
            instance_masks_dir_exists = dir_exists(instance_masks_dir)
            ann_dir = os.path.join(dataset_path, helper.ANNOTATION_DIR_NAME)
            if dir_exists(ann_dir):
                remove_junk_from_dir(ann_dir)
            if not machine_masks_dir_exists and dir_exists(ann_dir):
                if all([get_file_ext(d) == helper.MASK_EXT for d in os.listdir(ann_dir)]):
                    machine_masks_dir = ann_dir
                    machine_masks_dir_exists = True
            if not instance_masks_dir_exists and dir_exists(ann_dir):
                if all([dir_exists(os.path.join(ann_dir, d)) for d in os.listdir(ann_dir)]):
                    instance_masks_dir = ann_dir
                    instance_masks_dir_exists = True

            for img_name, img_path in zip(names, img_paths):
                item = self.Item(img_path)
                ann_data = {}
                machine_mask = os.path.join(machine_masks_dir, f"{img_name}{helper.MASK_EXT}")
                ann_detected = False
                if machine_masks_dir_exists and file_exists(machine_mask):
                    ann_data["machine_mask"] = machine_mask
                    ann_detected = True
                if instance_masks_dir_exists:
                    mask_dir = os.path.join(instance_masks_dir, img_name)
                    if dir_exists(mask_dir):
                        inst_masks = list_files(mask_dir, valid_extensions=[helper.MASK_EXT])
                        if len(inst_masks) > 0:
                            ann_detected = True
                            ann_data["inst_masks"] = inst_masks
                if ann_detected:
                    detected_ann_cnt += 1
                    item.ann_data = ann_data
                    self._items.append(item)

        return detected_ann_cnt > 0

    def to_supervisely(
        self,
        item: ImageConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> Annotation:
        """Convert to Supervisely format."""
        if meta is None:
            meta = self._meta

        ann = item.create_empty_annotation()
        if item.ann_data is None:
            return ann

        try:
            semantic_labels = []
            instance_labels = []
            semantic_mask_path = item.ann_data.get("machine_mask")
            instance_masks_paths = item.ann_data.get("inst_masks")
            if semantic_mask_path is not None:
                semantic_labels = helper.read_semantic_labels(
                    semantic_mask_path, self._classes_mapping, meta.obj_classes, renamed_classes
                )
            if instance_masks_paths is not None:
                instance_labels = helper.read_instance_labels(
                    instance_masks_paths, meta.obj_classes, renamed_classes
                )
            all_labels = validate_image_bounds(
                semantic_labels + instance_labels, Rectangle.from_size(item.shape)
            )

            ann = ann.add_labels(labels=all_labels)

            return ann
        except Exception as e:
            logger.warn(f"Failed to convert annotation: {repr(e)}")
            return ann
