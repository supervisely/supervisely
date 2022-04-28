# coding: utf-8


import os
import json
from typing import List, Union
import numpy as np
import SimpleITK as sitk

import pydicom
import stringcase
from supervisely.io.fs import get_file_ext
import supervisely.volume.nrrd_encoder as nrrd_encoder
from supervisely import logger

# Do NOT use directly for extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
ALLOWED_VOLUME_EXTENSIONS = [".nrrd", ".dcm"]


class UnsupportedVolumeFormat(Exception):
    pass


def get_extension(path: str):
    # magic.from_file("path", mime=True)
    # for nrrd:
    # application/octet-stream
    # for nifti(nii):
    # application/octet-stream
    # for dicom:
    # "application/dicom"

    ext = get_file_ext(path)
    if ext in ALLOWED_VOLUME_EXTENSIONS:
        return ext

    # case when dicom file does not have an extension
    import magic

    mime = magic.from_file(path, mime=True)
    if mime == "application/dicom":
        return ".dcm"
    return None


def is_valid_ext(ext: str) -> bool:
    return ext.lower() in ALLOWED_VOLUME_EXTENSIONS


def has_valid_ext(path: str) -> bool:
    return is_valid_ext(get_extension(path))


def validate_format(path: str):
    if not has_valid_ext(path):
        raise UnsupportedVolumeFormat(
            f"File {path} has unsupported volume extension. Supported extensions: {ALLOWED_VOLUME_EXTENSIONS}"
        )


def rescale_slope_intercept(value, slope, intercept):
    return value * slope + intercept


def normalize_volume_meta(meta):
    meta["intensity"]["min"] = rescale_slope_intercept(
        meta["intensity"]["min"],
        meta["rescaleSlope"],
        meta["rescaleIntercept"],
    )

    meta["intensity"]["max"] = rescale_slope_intercept(
        meta["intensity"]["max"],
        meta["rescaleSlope"],
        meta["rescaleIntercept"],
    )

    if "windowWidth" not in meta:
        meta["windowWidth"] = meta["intensity"]["max"] - meta["intensity"]["min"]

    if "windowCenter" not in meta:
        meta["windowCenter"] = meta["intensity"]["min"] + meta["windowWidth"] / 2

    return meta


def read_serie_volume_np(paths: List[str]) -> np.ndarray:
    sitk_volume, meta = read_serie_volume(paths)
    volume_np = sitk.GetArrayFromImage(sitk_volume)
    return volume_np, meta


_default_dicom_tags = [
    "SeriesInstanceUID",
    "Modality",
    "PatientID",
    "PatientName",
    "WindowCenter",
    "WindowWidth",
    "RescaleIntercept",
    "RescaleSlope",
    "PhotometricInterpretation",
]

_photometricInterpretationRGB = set(
    [
        "RGB",
        "PALETTE COLOR",
        "YBR_FULL",
        "YBR_FULL_422",
        "YBR_PARTIAL_422",
        "YBR_PARTIAL_420",
        "YBR_RCT",
    ]
)


def read_dicom_tags(path, allowed_keys: Union[None, List[str]] = _default_dicom_tags):

    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    vol_info = {}
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        tag = pydicom.tag.Tag(k.split("|")[0], k.split("|")[1])
        keyword = pydicom.datadict.keyword_for_tag(tag)
        keyword = stringcase.camelcase(keyword)
        if allowed_keys is not None and keyword not in allowed_keys:
            continue
        vol_info[keyword] = v
        if keyword in [
            "windowCenter",
            "windowWidth",
            "rescaleIntercept",
            "rescaleSlope",
        ]:
            vol_info[keyword] = float(vol_info[keyword].split("\\")[0])
        elif (
            keyword == "photometricInterpretation"
            and v in _photometricInterpretationRGB
        ):
            vol_info["channelsCount"] = 3
    return vol_info


def encode(volume_np: np.ndarray, volume_meta):
    volume_bytes = nrrd_encoder.encode(
        volume_np,
        header={
            "encoding": "gzip",
            "space": "RAS",
            "space directions": np.array(volume_meta["directions"])
            .reshape((3, 3))
            .tolist(),
            "space origin": volume_meta["origin"],
        },
        compression_level=1,
    )
    return volume_bytes


def inspect_series(root_dir: str):
    found_series = {}
    for d in os.walk(root_dir):
        dir = d[0]
        reader = sitk.ImageSeriesReader()
        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        series_found = reader.GetGDCMSeriesIDs(dir)
        sitk.ProcessObject_SetGlobalWarningDisplay(True)
        logger.info(f"Found {len(series_found)} series in directory {dir}")
        for serie in series_found:
            dicom_names = reader.GetGDCMSeriesFileNames(dir, serie)
            found_series[serie] = dicom_names
    logger.info(f"Total {len(found_series)} series in directory {root_dir}")
    return found_series


def read_serie_volume(paths):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(paths)
    sitk_volume = reader.Execute()

    sitk_volume = sitk.DICOMOrient(sitk_volume, "RAS")
    # RAS reorient image, does not work
    # orientation_filter = sitk.DICOMOrientImageFilter()
    # orientation_filter.SetDesiredCoordinateOrientation("RAS")
    # sitk_volume = orientation_filter.Execute(sitk_volume)

    # print("Output")
    # print("origin ", sitk_volume.GetOrigin())
    # print("direction ", sitk_volume.GetDirection())
    # print("spacing ", sitk_volume.GetSpacing())
    # print("size ", sitk_volume.GetSize())

    f_min_max = sitk.MinimumMaximumImageFilter()
    f_min_max.Execute(sitk_volume)

    meta = get_meta(
        sitk_volume.GetSize(),
        f_min_max.GetMinimum(),
        f_min_max.GetMaximum(),
        sitk_volume.GetSpacing(),
        sitk_volume.GetOrigin(),
        sitk_volume.GetDirection(),
        read_dicom_tags(paths[0]),
    )
    return sitk_volume, meta


def get_meta(
    sitk_shape, min_intensity, max_intensity, spacing, origin, directions, dicom_tags={}
):
    volume_meta = normalize_volume_meta(
        {
            "channelsCount": 1,
            "rescaleSlope": 1,
            "rescaleIntercept": 0,
            **dicom_tags,
            "intensity": {
                "min": min_intensity,
                "max": max_intensity,
            },
            "dimensionsIJK": {
                "x": sitk_shape[0],
                "y": sitk_shape[1],
                "z": sitk_shape[2],
            },
            "ACS": "RAS",
            # instead of IJK2WorldMatrix field
            "spacing": spacing,
            "origin": origin,
            "directions": directions,
            # "shape": sitk_shape,
        }
    )
    return volume_meta
