import re
from typing import Optional

import cv2
import numpy as np

from supervisely import Bitmap, Label, ObjClassCollection
from supervisely.io.fs import get_file_name

COLOR_MAP_FILE_NAME = "obj_class_to_machine_color.json"

MASK_EXT = ".png"
MATCH_ALL = "__all__"

IMAGE_DIR_NAME = "img"

ANNOTATION_DIR_NAME = "ann"
MASKS_MACHINE_DIR_NAME = "masks_machine"
MASKS_INSTANCE_DIR_NAME = "masks_instances"
MASKS_HUMAN_DIR_NAME = "masks_human"
MASK_DIRS = [
    ANNOTATION_DIR_NAME,
    MASKS_HUMAN_DIR_NAME,
    MASKS_INSTANCE_DIR_NAME,
    MASKS_MACHINE_DIR_NAME,
]

MARKERS = [IMAGE_DIR_NAME] + MASK_DIRS



def read_semantic_labels(
    mask_path: str,
    classes_mapping: dict,
    obj_classes: ObjClassCollection,
    renamed_classes: Optional[dict] = None,
) -> list:
    mask = cv2.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    labels_list = []
    for cls_name, color in classes_mapping.items():
        if color == MATCH_ALL:
            bool_mask = mask > 0
        elif isinstance(color, int):
            bool_mask = mask == color
        elif isinstance(color, list):
            bool_mask = np.isin(mask, color)
        else:
            raise ValueError(
                'Wrong color format. It must be integer, list of integers or special key string "__all__".'
            )

        if bool_mask.sum() == 0:
            continue

        if renamed_classes is not None:
            cls_name = renamed_classes.get(cls_name, cls_name)
        bitmap = Bitmap(data=bool_mask)
        obj_class = obj_classes.get(cls_name)
        labels_list.append(Label(geometry=bitmap, obj_class=obj_class))
    return labels_list


def read_instance_labels(
    mask_paths: list,
    obj_classes: list,
    renamed_classes: Optional[dict] = None,
) -> list:
    labels = []
    for instance_mask_path in mask_paths:
        cls_name = re.sub(r"_\d+", "", get_file_name(instance_mask_path))
        if renamed_classes is not None:
            cls_name = renamed_classes.get(cls_name, cls_name)
        obj_class = obj_classes.get(cls_name)
        bitmap = Bitmap.from_path(instance_mask_path)
        label = Label(geometry=bitmap, obj_class=obj_class)
        labels.append(label)
    return labels
