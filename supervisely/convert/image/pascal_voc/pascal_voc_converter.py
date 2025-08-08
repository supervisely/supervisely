import os
from collections import defaultdict
from typing import Dict, Optional, Set, Union

from supervisely import (
    Annotation,
    ProjectMeta,
    TagApplicableTo,
    TagMeta,
    TagValueType,
    logger,
)
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.image.pascal_voc import pascal_voc_helper
from supervisely.io.fs import (
    dir_exists,
    dirs_filter,
    file_exists,
    get_file_ext,
    get_file_name,
    list_files_recursively,
)
from supervisely.project.project_settings import LabelingInterface


class PascalVOCConverter(ImageConverter):
    class Item(ImageConverter.Item):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self._segm_path = None
            self._inst_path = None

        @property
        def segm_path(self) -> str:
            return self._segm_path

        @property
        def inst_path(self) -> str:
            return self._inst_path

        @segm_path.setter
        def segm_path(self, segm_path: str) -> None:
            self._segm_path = segm_path

        @inst_path.setter
        def inst_path(self, inst_path: str) -> None:
            self._inst_path = inst_path

    def __init__(
        self,
        input_data: str,
        labeling_interface: Optional[Union[LabelingInterface, str]],
        upload_as_links: bool,
        remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)

        self.color2class_name: Optional[Dict[str, str]] = None
        self.with_instances: bool = False
        self._imgs_dir: Optional[str] = None
        self._segm_dir: Optional[str] = None
        self._inst_dir: Optional[str] = None
        self._bbox_classes_map: Dict[str, str] = {}

    def __str__(self) -> str:
        return AvailableImageConverters.PASCAL_VOC

    @property
    def ann_ext(self) -> str:
        return ".xml"

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta) -> bool:
        # TODO: implement annotation validation
        pass

    def validate_format(self) -> bool:
        xml_files_exist = self._scan_for_xml_files()
        possible_pascal_voc_dir = self._scan_for_segm_dirs()
        if not possible_pascal_voc_dir and not xml_files_exist:
            return False
        if not possible_pascal_voc_dir:
            self._imgs_dir = self._input_data
            possible_pascal_voc_dir = self._input_data

        self._meta = self._generate_meta(possible_pascal_voc_dir)
        detected_ann_cnt = self._create_items(possible_pascal_voc_dir)
        return detected_ann_cnt > 0

    def _scan_for_xml_files(self) -> bool:
        for _, _, file_names in os.walk(self._input_data):
            for filename in file_names:
                if get_file_ext(filename) == self.ann_ext:
                    return True
        return False

    def _scan_for_segm_dirs(self) -> Optional[str]:
        def check_function(dir_path: str) -> bool:
            possible_image_dirs = ["JPEGImages", "Images", "images", "imgs", "img"]
            possible_segm_dirs = [
                "SegmentationClass",
                "segmentation",
                "segmentations",
                "Segmentation",
                "Segmentations",
                "masks",
                "segm",
            ]
            possible_inst_dirs = ["SegmentationObject", "instances", "objects"]

            if not any([dir_exists(os.path.join(dir_path, p)) for p in possible_image_dirs]):
                return False
            if not any([dir_exists(os.path.join(dir_path, p)) for p in possible_segm_dirs]):
                return False
            for d in possible_image_dirs:
                if dir_exists(os.path.join(dir_path, d)):
                    self._imgs_dir = os.path.join(dir_path, d)
                    break
            for d in possible_segm_dirs:
                if dir_exists(os.path.join(dir_path, d)):
                    self._segm_dir = os.path.join(dir_path, d)
                    break
            for d in possible_inst_dirs:
                if dir_exists(os.path.join(dir_path, d)):
                    self._inst_dir = os.path.join(dir_path, d)
                    break

            return self._imgs_dir is not None and self._segm_dir is not None

        possible_pascal_voc_dir = [d for d in dirs_filter(self._input_data, check_function)]
        if len(possible_pascal_voc_dir) > 1:
            logger.warn("Multiple Pascal VOC directories not supported")
            return
        elif len(possible_pascal_voc_dir) == 0:
            return
        else:
            return possible_pascal_voc_dir[0]

    def _generate_meta(self, possible_pascal_voc_dir: str) -> ProjectMeta:
        colors_file = os.path.join(possible_pascal_voc_dir, "colors.txt")
        obj_classes, self.color2class_name = pascal_voc_helper.read_colors(colors_file)
        return ProjectMeta(obj_classes=obj_classes)

    def _create_items(self, possible_pascal_voc_dir: str) -> int:
        existing_cls_names = set([cls.name for cls in self._meta.obj_classes])
        tags_to_values = defaultdict(set)
        detected_ann_cnt = 0

        images_list = list_files_recursively(self._imgs_dir, valid_extensions=self.allowed_exts)
        img_ann_map = {
            get_file_name(path): path
            for path in list_files_recursively(possible_pascal_voc_dir, [self.ann_ext])
        }

        self._items = []
        for image_path in images_list:
            item = self.Item(image_path)
            item_name_noext = get_file_name(item.name)
            item = self._scan_for_item_segm_paths(item, item_name_noext)
            ann_path = img_ann_map.get(item_name_noext) or img_ann_map.get(item.name)
            item = self._scan_for_item_ann_path_and_update_meta(
                item, ann_path, existing_cls_names, tags_to_values
            )

            if item.ann_data or item.segm_path:
                detected_ann_cnt += 1
            self._items.append(item)
        self._meta = self._update_meta_with_tags(tags_to_values)
        return detected_ann_cnt

    def _update_meta_with_tags(self, tags_to_values: Dict[str, Set[str]]) -> ProjectMeta:
        meta = self._meta
        object_class_names = set(meta.obj_classes.keys())
        for tag_name, values in tags_to_values.items():
            tag_meta = meta.get_tag_meta(tag_name)
            if tag_meta is not None:
                continue
            if tag_name in pascal_voc_helper.DEFAULT_SUBCLASSES:
                if values.difference({"0", "1"}):
                    logger.warning(
                        f"Tag '{tag_name}' has non-binary values.", extra={"values": values}
                    )
                tag_meta = TagMeta(tag_name, TagValueType.NONE)
            elif tag_name in object_class_names:
                tag_meta = TagMeta(
                    tag_name,
                    TagValueType.ONEOF_STRING,
                    possible_values=list(values),
                    applicable_to=TagApplicableTo.OBJECTS_ONLY,
                    applicable_classes=[tag_name],
                )
            else:
                tag_meta = TagMeta(tag_name, TagValueType.ANY_STRING)
            meta = meta.add_tag_meta(tag_meta)
        return meta

    def _scan_for_item_segm_paths(self, item: Item, item_name_noext: str) -> Item:
        if self._segm_dir is not None:
            segm_path = os.path.join(self._segm_dir, f"{item_name_noext}.png")
            if file_exists(segm_path):
                item.segm_path = segm_path
        if self._inst_dir is not None:
            inst_path = os.path.join(self._inst_dir, f"{item_name_noext}.png")
            if file_exists(inst_path):
                item.inst_path = inst_path

        return item

    def _scan_for_item_ann_path_and_update_meta(
        self,
        item: Item,
        ann_path: Optional[str],
        existing_cls_names: Set[str],
        tags_to_values: Dict[str, Set[str]],
    ) -> Item:
        if ann_path is None:
            return item
        if not file_exists(ann_path):
            return item
        self._meta = pascal_voc_helper.update_meta_from_xml(
            ann_path, self._meta, existing_cls_names, self._bbox_classes_map, tags_to_values
        )
        item.ann_data = ann_path
        return item

    def to_supervisely(
        self,
        item: Item,
        meta: Optional[ProjectMeta] = None,
        renamed_classes: Optional[Dict[str, str]] = None,
        renamed_tags: Optional[Dict[str, str]] = None,
    ) -> Annotation:
        if meta is None:
            meta = self._meta

        try:
            item.set_shape()
            return pascal_voc_helper.get_ann(
                item,
                self.color2class_name,
                meta,
                self._bbox_classes_map,
                renamed_classes,
                renamed_tags,
            )
        except Exception as e:
            logger.warn(f"Failed to convert annotation: {repr(e)}")
            return item.create_empty_annotation()
