import os
from typing import List

from supervisely import Annotation, ProjectMeta, logger
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.image.pascal_voc import pascal_voc_helper
from supervisely.imaging.image import SUPPORTED_IMG_EXTS
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

        def set_segm_path(self, segm_path: str) -> None:
            self._segm_path = segm_path

        def set_inst_path(self, inst_path: str) -> None:
            self._inst_path = inst_path

    def __init__(self, input_data: str):
        self._input_data: str = input_data
        self._items: List[ImageConverter.Item] = []
        self._meta: ProjectMeta = None
        self.color2class_name = None
        self.with_instances = False

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
            if not dir_exists(os.path.join(dir_path, "ImageSets")):
                return False
            if not dir_exists(os.path.join(dir_path, "ImageSets", "Segmentation")):
                return False
            if not dir_exists(os.path.join(dir_path, "JPEGImages")):
                return False
            if not dir_exists(os.path.join(dir_path, "SegmentationClass")):
                return False
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
        self.with_instances = dir_exists(
            os.path.join(possible_pascal_voc_dir, "SegmentationObject")
        )
        self._meta = ProjectMeta(obj_classes=obj_classes)

        images_list = list_files_recursively(
            os.path.join(possible_pascal_voc_dir, "JPEGImages"),
            valid_extensions=SUPPORTED_IMG_EXTS,
        )

        # create Items
        self._items = []
        for image_path in images_list:
            item = self.Item(image_path)
            item_name_noext = get_file_name(item.name)
            segm_path = os.path.join(
                possible_pascal_voc_dir, "SegmentationClass", item_name_noext + ".png"
            )
            if file_exists(segm_path):
                item.set_segm_path(segm_path)
                detected_ann_cnt += 1
            if self.with_instances:
                inst_path = os.path.join(
                    possible_pascal_voc_dir, "SegmentationObject", item_name_noext + ".png"
                )
                if file_exists(inst_path):
                    item.set_inst_path(inst_path)
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
            ann = pascal_voc_helper.get_ann(item, self.color2class_name, renamed_classes)
            return ann

        except Exception as e:
            logger.warn(f"Failed to convert annotation: {repr(e)}")
            return item.create_empty_annotation()
