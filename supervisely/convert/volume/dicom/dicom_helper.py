import os
from typing import List

import nrrd
import numpy as np
from pydicom import FileDataset
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


def read_and_convert_to_monochrome2(dcm_path: str):
    import pydicom

    try:
        dcm = pydicom.dcmread(dcm_path)
    except Exception as e:
        logger.warning("Failed to read DICOM file: " + str(e))
        return

    try:
        if dcm.file_meta.TransferSyntaxUID.is_compressed:
            dcm.decompress()
    except Exception as e:
        logger.warning("Failed to decompress DICOM file: " + str(e))
        return

    convert_to_monochrome2(dcm_path, dcm)


def convert_to_monochrome2(dcm_path: str, dcm: FileDataset) -> FileDataset:
    if getattr(dcm, "PhotometricInterpretation", None) == "YBR_FULL_422":
        # * Convert dicom to monochrome
        monochrome = None
        pixel_array = dcm.pixel_array

        if len(pixel_array.shape) == 4 and pixel_array.shape[-1] == 3:
            monochrome = pixel_array[..., 0].astype(np.uint8)
        elif len(pixel_array.shape) == 3 and pixel_array.shape[-1] == 3:
            monochrome = pixel_array[..., 0].astype(np.uint8)
        else:
            logger.warning(
                "Unexpected shape for YBR_FULL_422 data: " + str(pixel_array.shape)
            )
            return dcm

        logger.debug("Monochrome shape: " + str(monochrome.shape))

        try:
            dcm.SamplesPerPixel = 1
            dcm.PhotometricInterpretation = "MONOCHROME2"
            dcm.PlanarConfiguration = 0
            if len(monochrome.shape) == 3:
                dcm.NumberOfFrames = str(monochrome.shape[0])
                dcm.Rows, dcm.Columns = monochrome.shape[1:3]
            elif len(monochrome.shape) == 2:
                dcm.Rows, dcm.Columns = monochrome.shape[0:2]
                if hasattr(dcm, "NumberOfFrames"):
                    delattr(dcm, "NumberOfFrames")
            dcm.PixelData = monochrome.tobytes()
        except AttributeError as ae:
            logger.error(f"Error occurred while converting dicom to monochrome: {ae}")
            return dcm

        logger.info("Rewriting DICOM file with monochrome2 format")
        dcm.save_as(dcm_path)
    return dcm
