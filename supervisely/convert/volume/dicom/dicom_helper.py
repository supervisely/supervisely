import os
from typing import List

import nrrd
import numpy as np
import pydicom
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


# pydicom 3 (Python >= 3.10) removed read_file/write_file and changed what Dataset.decompress() and
# Dataset.pixel_array do with colour data. The helpers below give the same result on pydicom 2 and 3.
PYDICOM_3 = int(pydicom.__version__.split(".", 1)[0]) >= 3


def decompress(dcm: FileDataset) -> FileDataset:
    """Decompress a dataset in place the way pydicom 2 did, on pydicom 2 and 3.

    pydicom 3 changed two defaults of :meth:`Dataset.decompress`: it converts YBR colour data to RGB
    (``as_rgb=True``), which hides the YBR_FULL_422 data that :func:`convert_to_monochrome2` turns
    into a monochrome image, and it replaces the SOP Instance UID with a new one
    (``generate_instance_uid=True``), which then ends up in the image metadata. Both are switched off.
    """
    if PYDICOM_3:
        dcm.decompress(as_rgb=False, generate_instance_uid=False)
    else:
        dcm.decompress()
    return dcm


def _ybr_pixel_array(dcm: FileDataset) -> np.ndarray:
    """Pixel data of a YBR_FULL_422 dataset as YBR (Y in channel 0), not RGB, on pydicom 2 and 3.

    pydicom 3 converts YBR to RGB by default, and after :func:`decompress` the data is no longer
    subsampled although the Photometric Interpretation still says YBR_FULL_422, which pydicom 3
    rejects; pydicom 2 returned the decoded YBR array in both cases.
    """
    if not PYDICOM_3:
        return dcm.pixel_array

    from pydicom.pixels import pixel_array

    kwargs = {"as_rgb": False}
    transfer_syntax = dcm.get("file_meta", {}).get("TransferSyntaxUID")
    if transfer_syntax is not None and not pydicom.uid.UID(transfer_syntax).is_compressed:
        frames = int(dcm.get("NumberOfFrames", 1) or 1)
        full_size = dcm.Rows * dcm.Columns * frames * 3 * ((dcm.BitsAllocated + 7) // 8)
        if len(dcm.PixelData) >= full_size:
            kwargs["photometric_interpretation"] = "YBR_FULL"
    return pixel_array(dcm, **kwargs)


def read_and_convert_to_monochrome2(dcm_path: str):
    try:
        dcm = pydicom.dcmread(dcm_path)
    except Exception as e:
        logger.warning("Failed to read DICOM file: " + str(e))
        return

    try:
        if dcm.file_meta.TransferSyntaxUID.is_compressed:
            decompress(dcm)
    except Exception as e:
        logger.warning("Failed to decompress DICOM file: " + str(e))
        return

    convert_to_monochrome2(dcm_path, dcm)


def convert_to_monochrome2(dcm_path: str, dcm: FileDataset) -> FileDataset:
    if getattr(dcm, "PhotometricInterpretation", None) == "YBR_FULL_422":
        # * Convert dicom to monochrome
        monochrome = None
        pixel_array = _ybr_pixel_array(dcm)

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
