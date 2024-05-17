import os
from typing import List

import nrrd
import numpy as np

from supervisely import logger
from supervisely.io.fs import file_exists
from supervisely.volume.volume import read_dicom_serie_volume_np


def dcm_to_nrrd(id: str, paths: List[str]) -> str:
    """Convert DICOM series to NRRD format."""

    volume_np, volume_meta = read_dicom_serie_volume_np(paths)
    logger.info(f"PatientID and PatientName field for {id} will be anonymized.")
    serie_dir = os.path.dirname(paths[0])
    parent_dir = os.path.dirname(serie_dir)
    nrrd_path = os.path.join(parent_dir, f"{id}.nrrd")
    i = 0
    while file_exists(nrrd_path):
        i += 1
        nrrd_path = os.path.join(parent_dir, f"{id}_{i}.nrrd")

    nrrd_header = {
        "space": volume_meta.get("ACS", "RAS"),
    }
    if volume_meta.get("origin", None) is not None:
        nrrd_header["space origin"] = volume_meta.get("origin")
    directions = volume_meta.get("directions", None)
    spacing = volume_meta.get("spacing", None)
    if directions is not None and spacing is not None:
        directions = np.array(directions).reshape(3, 3)
        directions *= spacing
        nrrd_header["space directions"] = directions

    nrrd.write(nrrd_path, volume_np, nrrd_header)

    return nrrd_path, volume_meta
