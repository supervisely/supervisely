import os
from typing import List

from supervisely import Annotation, ProjectMeta, logger
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.image.pascal_voc import pascal_voc_helper
from supervisely.io.fs import (
    dir_exists,
    dirs_filter,
    file_exists,
    get_file_name,
    list_files_recursively,
    remove_junk_from_dir,
)


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

    def __init__(self, input_data: str, labeling_interface: str) -> None:
        self._input_data: str = input_data
        self._items: List[ImageConverter.Item] = []
        self._meta: ProjectMeta = None
        self.color2class_name = None
        self.with_instances = False
        self._imgs_dir = None
        self._segm_dir = None
        self._inst_dir = None
        self._labeling_interface = labeling_interface
        self._bbox_classes_map = {}

    def __str__(self) -> str:
        return AvailableImageConverters.PASCAL_VOC

    @property
    def ann_ext(self) -> str:
        return ".xml"

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta) -> bool:
        pass

    def validate_format(self) -> bool:
        detected_ann_cnt = 0

        def check_function(dir_path):
            possible_image_dir_names = ["JPEGImages", "Images", "images", "imgs", "img"]
            possible_segm_dir_names = [
                "SegmentationClass",
                "segmentation",
                "segmentations",
                "Segmentation",
                "Segmentations",
                "masks",
                "segm",
            ]
            if not any([dir_exists(os.path.join(dir_path, p)) for p in possible_image_dir_names]):
                return False
            if not any([dir_exists(os.path.join(dir_path, p)) for p in possible_segm_dir_names]):
                return False
            for d in possible_image_dir_names:
                if dir_exists(os.path.join(dir_path, d)):
                    self._imgs_dir = os.path.join(dir_path, d)
                    break
            for d in possible_segm_dir_names:
                if dir_exists(os.path.join(dir_path, d)):
                    self._segm_dir = os.path.join(dir_path, d)
                    break
            return True

        possible_pascal_voc_dir = [d for d in dirs_filter(self._input_data, check_function)]
        if len(possible_pascal_voc_dir) == 0:
            return False
        if len(possible_pascal_voc_dir) > 1:
            logger.warn("Multiple Pascal VOC directories not supported")
            return False

        possible_pascal_voc_dir = possible_pascal_voc_dir[0]
        remove_junk_from_dir(possible_pascal_voc_dir)

        colors_file = os.path.join(possible_pascal_voc_dir, "colors.txt")
        obj_classes, color2class_name = pascal_voc_helper.read_colors(colors_file)
        self.color2class_name = color2class_name
        for p in ["SegmentationObject", "instances", "objects"]:
            self.with_instances = dir_exists(os.path.join(possible_pascal_voc_dir, p))
            if self.with_instances:
                self._inst_dir = os.path.join(possible_pascal_voc_dir, p)
                break
        self._meta = ProjectMeta(obj_classes=obj_classes)

        # list all images and collect xml annotations
        images_list = list_files_recursively(self._imgs_dir, valid_extensions=self.allowed_exts)
        img_ann_map = {}
        for path in list_files_recursively(possible_pascal_voc_dir, valid_extensions=[".xml"]):
            img_ann_map[get_file_name(path)] = path
        existing_cls_names = set([cls.name for cls in self._meta.obj_classes])

        # create Items
        self._items = []
        for image_path in images_list:
            item = self.Item(image_path)
            item_name_noext = get_file_name(item.name)
            segm_path = os.path.join(self._segm_dir, item_name_noext + ".png")
            if file_exists(segm_path):
                item.segm_path = segm_path
                detected_ann_cnt += 1
            if self.with_instances:
                inst_path = os.path.join(self._inst_dir, item_name_noext + ".png")
                if file_exists(inst_path):
                    item.inst_path = inst_path
            ann_path = img_ann_map.get(item_name_noext)
            if ann_path is None:
                ann_path = img_ann_map.get(item.name)
            if ann_path is not None and file_exists(ann_path):
                self._meta = pascal_voc_helper.update_meta_from_xml(
                    ann_path, self._meta, existing_cls_names, self._bbox_classes_map
                )
                item.ann_data = ann_path
            self._items.append(item)
        return detected_ann_cnt > 0

    def to_supervisely(
        self,
        item: ImageConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> Annotation:
        if meta is None:
            meta = self._meta

        try:
            item.set_shape()
            ann = pascal_voc_helper.get_ann(
                item, self.color2class_name, meta, self._bbox_classes_map, renamed_classes
            )
            return ann

        except Exception as e:
            logger.warn(f"Failed to convert annotation: {repr(e)}")
            return item.create_empty_annotation()
