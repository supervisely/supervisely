import os
from typing import Generator

import nrrd
import numpy as np

from supervisely.collection.str_enum import StrEnum
from supervisely.geometry.mask_3d import Mask3D
from supervisely.io.fs import ensure_base_path, get_file_ext, get_file_name
from supervisely.sly_logger import logger
from supervisely.volume.volume import convert_3d_nifti_to_nrrd

VOLUME_NAME = "anatomic"
LABEL_NAME = ["inference", "label", "annotation", "mask", "segmentation"]


class PlanePrefix(str, StrEnum):
    """Prefix for plane names."""

    CORONAL = "cor"
    SAGITTAL = "sag"
    AXIAL = "axl"


def read_cls_color_map(path: str) -> dict:
    """Read class color map from TXT file.

    ```txt
    1 Femur 255 0 0
    2 Femoral cartilage 0 255 0
    3 Tibia 0 0 255
    4 Tibia cartilage 255 255 0
    5 Patella 0 255 255
    6 Patellar cartilage 255 0 255
    7 Miniscus 175 175 175
    ```
    """

    cls_color_map = {}
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as file:
            for line in file:
                parts = line.strip().split()
                color = None
                try:
                    color = list(map(int, parts[-3:]))
                except:
                    pass
                cls_id = int(parts[0])
                if not color:
                    cls_name = " ".join(parts[1:])
                else:
                    cls_name = " ".join(parts[1:-3])
                if cls_id in cls_color_map:
                    logger.warning(f"Duplicate class ID {cls_id} found in color map.")
                if cls_name in cls_color_map:
                    logger.warning(f"Duplicate class name {cls_name} found in color map.")
                if len(color) != 3:
                    logger.warning(f"Invalid color format for class {cls_name}. Expected 3 values.")
                if any(c < 0 or c > 255 for c in color):
                    logger.warning(
                        f"Invalid color value for class {cls_name}. Expected values between 0 and 255."
                    )
                cls_color_map[cls_id] = (cls_name, color)
    except Exception as e:
        logger.warning(f"Failed to read class color map from {path}: {e}")
        return None
    return cls_color_map


def nifti_to_nrrd(nii_file_path: str, converted_dir: str) -> str:
    """Convert NIfTI 3D volume file to NRRD 3D volume file."""

    output_name = get_file_name(nii_file_path)
    if get_file_ext(output_name) == ".nii":
        output_name = get_file_name(output_name)

    data, header = convert_3d_nifti_to_nrrd(nii_file_path)

    nrrd_file_path = os.path.join(converted_dir, f"{output_name}.nrrd")
    ensure_base_path(nrrd_file_path)

    nrrd.write(nrrd_file_path, data, header)
    return nrrd_file_path


def get_annotation_from_nii(path: str) -> Generator[Mask3D, None, None]:
    """Get annotation from NIfTI 3D volume file."""

    data, _ = convert_3d_nifti_to_nrrd(path)
    unique_classes = np.unique(data)

    for class_id in unique_classes:
        if class_id == 0:
            continue
        mask = Mask3D(data == class_id)
        yield mask, class_id
