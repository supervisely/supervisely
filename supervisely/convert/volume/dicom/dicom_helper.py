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

def convert_to_monochrome2(dcm_path: str):
    import pydicom

    is_modified = False

    try:
        dcm = pydicom.dcmread(dcm_path)
    except Exception as e:
        logger.warn("Failed to read DICOM file: " + str(e))
        return

    try:
        if dcm.file_meta.TransferSyntaxUID.is_compressed:
            dcm.decompress()
            is_modified = True
    except Exception as e:
        logger.warn("Failed to decompress DICOM file: " + str(e))
        return

    if getattr(dcm, "PhotometricInterpretation", None) == "YBR_FULL_422":
        # * Convert dicom to monochrome
        if len(dcm.pixel_array.shape) == 4 and dcm.pixel_array.shape[-1] == 3:
            monochrome = dcm.pixel_array[..., 0].astype(np.uint8)
        else:
            logger.warn("Unexpected shape for YBR_FULL_422 data: " + str(dcm.pixel_array.shape))

        try:
            dcm.SamplesPerPixel = 1
            dcm.PhotometricInterpretation = "MONOCHROME2"
            dcm.PlanarConfiguration = 0
            if len(monochrome.shape) == 3:
                dcm.NumberOfFrames = str(monochrome.shape[0])
                dcm.Rows, dcm.Columns = monochrome.shape[1:3]
            dcm.PixelData = monochrome.tobytes()
        except AttributeError as ae:
            logger.error(f"Error occurred while converting dicom to monochrome: {ae}")

        logger.info("Rewriting DICOM file with MONOCHROME2 photometric interpretation.")
        is_modified = True

    try:
        if is_modified:
            dcm.save_as(dcm_path)
    except Exception as e:
        logger.warn("Failed to save DICOM file: " + str(e))
