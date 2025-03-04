import os
from typing import Generator

import nrrd
import numpy as np

from supervisely.geometry.mask_3d import Mask3D
from supervisely.io.fs import ensure_base_path, get_file_ext, get_file_name
from supervisely.volume.volume import convert_3d_nifti_to_nrrd


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
