import os
from pathlib import Path
from typing import Dict, Optional, Union

import supervisely.convert.image.cityscapes.cityscapes_helper as helper
from supervisely import (
    Annotation,
    ObjClass,
    ObjClassCollection,
    Polygon,
    ProjectMeta,
    Tag,
    TagMeta,
    TagValueType,
    logger,
)
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.imaging.color import generate_rgb
from supervisely.io.fs import JUNK_FILES, get_file_name
from supervisely.io.json import load_json_file
from supervisely.project.project_settings import LabelingInterface


class CityscapesConverter(ImageConverter):

    def __init__(
            self,
            input_data: str,
            labeling_interface: Optional[Union[LabelingInterface, str]],
            upload_as_links: bool,
            remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)

        self._classes_mapping = {}
        self._supports_links = True
        self._force_shape_for_links = self.upload_as_links

    def __str__(self):
        return AvailableImageConverters.CITYSCAPES

    @property
    def key_file_ext(self) -> str:
        return ".json"
    
    @property
    def ann_ext(self) -> str:
        return ".json"

    def ann_file_ext(self) -> str:
        return ".json"

    def validate_ann_file(self, ann_file_path: str) -> bool:
        try:
            ann_json = load_json_file(ann_file_path)
            if isinstance(ann_json, dict):
                objects = ann_json.get("objects")
                if objects is not None:
                    return True
                else:
                    logger.warn(f"Couldn't read objects from annoation file: '{ann_file_path}'")
                    return False
            else:
                return False
        except:
            logger.warn(f"Failed to read annotation file: '{ann_file_path}'")
            return False

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            classes_mapping = load_json_file(key_file_path)
            if isinstance(classes_mapping, list):
                obj_classes = []
                for obj_class_json in classes_mapping:
                    if (
                        obj_class_json.get(
                            "name",
                        )
                        is None
                    ):
                        continue
                    obj_class_name = obj_class_json["name"]
                    if helper.CITYSCAPES_CLASSES_TO_COLORS_MAP.get(obj_class_name, None):
                        obj_class = ObjClass(
                            name=obj_class_name,
                            geometry_type=Polygon,
                            color=helper.CITYSCAPES_CLASSES_TO_COLORS_MAP[obj_class_name],
                        )
                    else:
                        if obj_class_json.get("color") is None:
                            new_color = obj_class_json["color"]
                        else:
                            new_color = generate_rgb(helper.CITYSCAPES_COLORS)
                        helper.CITYSCAPES_COLORS.append(new_color)
                        obj_class = ObjClass(
                            name=obj_class_name, geometry_type=Polygon, color=new_color
                        )
                    obj_classes.append(obj_class)
                obj_class_collection = ObjClassCollection(obj_classes)
                self._classes_mapping = classes_mapping
                tag_meta = TagMeta("split", TagValueType.ANY_STRING)
                if self._meta is None:
                    self._meta = ProjectMeta(obj_classes=obj_class_collection, tag_metas=[tag_meta])
                else:
                    if not self._meta.tag_metas.has_key("split"):
                        self._meta = self._meta.add_tag_meta(tag_meta)
                    self._meta = self._meta.add_obj_classes(obj_classes)
                return True
            else:
                return False
        except Exception as e:
            logger.warn(f"Failed to read 'class_to_id.json': {repr(e)}")
            return False

    def validate_format(self) -> bool:
        if self.upload_as_links:
            self._download_remote_ann_files()
        detected_ann_cnt = 0
        images_list, ann_dict = [], {}
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                dir_name = os.path.basename(root)
                file_name = get_file_name(full_path)
                if file.lower() == "class_to_id.json":
                    success = self.validate_key_file(os.path.join(root, file))
                    if not success:
                        logger.warn(
                            f"Failed to validate key file: '{file}'. Will use default cityscapes classes."
                        )
                if file in JUNK_FILES:
                    continue
                if file.endswith("_gtFine_polygons.json"):
                    success = self.validate_ann_file(full_path)
                    if success:
                        detected_ann_cnt += 1
                        ann_dict[file] = full_path
                if file_name.endswith("_leftImg8bit"):
                    if self.is_image(full_path):
                        images_list.append(full_path)

        meta = self._meta if self._meta is not None else ProjectMeta()
        self._items = []
        for image_path in images_list:
            image_name = get_file_name(image_path)
            if image_name.endswith("_leftImg8bit"):
                image_name = image_name[: -len("_leftImg8bit")]
            item = self.Item(image_path)
            ann_path = ann_dict.get(f"{image_name}_gtFine_polygons.json")
            if ann_path is not None:
                meta = self._generate_meta_from_ann(ann_path, meta)
                item.ann_data = ann_path
            self._items.append(item)

        if detected_ann_cnt > 0:
            tag_meta = TagMeta("split", TagValueType.ANY_STRING)
            if not meta.tag_metas.has_key(tag_meta.name):
                meta = meta.add_tag_meta(tag_meta)
            self._meta = meta
        return detected_ann_cnt > 0

    def _generate_meta_from_ann(self, ann_path: str, meta: ProjectMeta) -> ProjectMeta:
        ann_data = load_json_file(ann_path)
        for obj in ann_data["objects"]:
            class_name = obj["label"]
            obj_class = meta.get_obj_class(class_name)
            if obj_class is None:
                obj_class = ObjClass(class_name, Polygon)
                meta = meta.add_obj_class(obj_class)
        return meta

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

        split_tag_name = renamed_tags.get("split", "split") if renamed_tags is not None else "split"
        tag_meta = meta.get_tag_meta(split_tag_name)
        path_parts = [ppart.lower() for ppart in Path(item.path).parts]
        if helper.VAL_TAG in path_parts:
            tag_value = helper.VAL_TAG
        elif helper.TEST_TAG in path_parts:
            tag_value = helper.TEST_TAG
        else:
            tag_value = helper.TRAIN_TAG
        split_tag = Tag(tag_meta, tag_value)

        ann = ann.add_tag(split_tag)

        if item.ann_data is None:
            return ann
        ann_path = item.ann_data
        try:
            if ann_path is not None:
                ann = helper.create_ann_from_file(ann, ann_path, meta, renamed_classes)
            return ann
        except Exception as e:
            logger.warn(f"Failed to convert annotation: {repr(e)}")
            return ann
