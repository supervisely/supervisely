import os
from typing import List

import nrrd

from supervisely.volume.volume import read_dicom_serie_volume_np
from supervisely import logger
from supervisely.io.fs import file_exists


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
    nrrd.write(nrrd_path, volume_np, volume_meta)

    return nrrd_path, volume_meta
