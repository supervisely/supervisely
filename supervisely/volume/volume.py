# coding: utf-8


import os
import json
from typing import List, Union
import numpy as np

import pydicom
import stringcase
from supervisely.io.fs import get_file_ext, list_files_recursively, list_files
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


def read_dicom_serie_volume_np(paths: List[str], anonymize=True) -> np.ndarray:
    import SimpleITK as sitk
    sitk_volume, meta = read_dicom_serie_volume(paths, anonymize=anonymize)
    # for debug:
    # sitk.WriteImage(sitk_volume, "/work/output/sitk.nrrd", useCompression=False, compressionLevel=9)
    # with open("/work/output/test.nrrd", "wb") as file:
    #     file.write(b)
    volume_np = sitk.GetArrayFromImage(sitk_volume)
    volume_np = np.transpose(volume_np, (2, 1, 0))
    return volume_np, meta


_anonymize_tags = ["PatientID", "PatientName"]
_default_dicom_tags = [
    "SeriesInstanceUID",
    "Modality",
    "WindowCenter",
    "WindowWidth",
    "RescaleIntercept",
    "RescaleSlope",
    "PhotometricInterpretation",
]
_default_dicom_tags.extend(_anonymize_tags)

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


def read_dicom_tags(
    path, allowed_keys: Union[None, List[str]] = _default_dicom_tags, anonymize=True
):
    import SimpleITK as sitk
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    vol_info = {}
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        tag = pydicom.tag.Tag(k.split("|")[0], k.split("|")[1])
        keyword = pydicom.datadict.keyword_for_tag(tag)
        if allowed_keys is not None and keyword not in allowed_keys:
            continue
        if anonymize is True and keyword in _anonymize_tags:
            v = "anonymized"
        keyword = stringcase.camelcase(keyword)
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
    directions = np.array(volume_meta["directions"]).reshape(3, 3)
    directions *= volume_meta["spacing"]

    volume_bytes = nrrd_encoder.encode(
        volume_np,
        header={
            "encoding": "gzip",
            # "space": "left-posterior-superior",
            "space": "right-anterior-superior",
            "space directions": directions.T.tolist(),
            "space origin": volume_meta["origin"],
        },
        compression_level=1,
    )

    # with open("/work/output/test.nrrd", "wb") as file:
    #     file.write(volume_bytes)

    return volume_bytes


def inspect_dicom_series(root_dir: str):
    import SimpleITK as sitk
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


def _sitk_image_orient_ras(sitk_volume):
    import SimpleITK as sitk
    if sitk_volume.GetDimension() == 4 and sitk_volume.GetSize()[3] == 1:
        sitk_volume = sitk_volume[:, :, :, 0]

    sitk_volume = sitk.DICOMOrient(sitk_volume, "RAS")
    # RAS reorient image using filter
    # orientation_filter = sitk.DICOMOrientImageFilter()
    # orientation_filter.SetDesiredCoordinateOrientation("RAS")
    # sitk_volume = orientation_filter.Execute(sitk_volume)

    # https://discourse.itk.org/t/getdirection-and-getorigin-for-simpleitk-c-implementation/3472/8
    origin = list(sitk_volume.GetOrigin())
    directions = list(sitk_volume.GetDirection())
    origin[0] *= -1
    origin[1] *= -1
    directions[0] *= -1
    directions[1] *= -1
    directions[3] *= -1
    directions[4] *= -1
    directions[6] *= -1
    directions[7] *= -1
    sitk_volume.SetOrigin(origin)
    sitk_volume.SetDirection(directions)
    return sitk_volume


def read_dicom_serie_volume(paths, anonymize=True):
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(paths)
    sitk_volume = reader.Execute()
    sitk_volume = _sitk_image_orient_ras(sitk_volume)
    dicom_tags = read_dicom_tags(paths[0], anonymize=anonymize)

    f_min_max = sitk.MinimumMaximumImageFilter()
    f_min_max.Execute(sitk_volume)
    meta = get_meta(
        sitk_volume.GetSize(),
        f_min_max.GetMinimum(),
        f_min_max.GetMaximum(),
        sitk_volume.GetSpacing(),
        sitk_volume.GetOrigin(),
        sitk_volume.GetDirection(),
        dicom_tags,
    )
    return sitk_volume, meta


def compose_ijk_2_world_mat(spacing, origin, directions):
    mat = np.eye(4)
    mat[:3, :3] = (np.array(directions).reshape(3, 3) * spacing).T
    mat[:3, 3] = origin
    return mat


def get_meta(
    sitk_shape, min_intensity, max_intensity, spacing, origin, directions, dicom_tags={}
):

    # x = 1 - sagittal
    # y = 1 - coronal
    # z = 1 - axial
    volume_meta = normalize_volume_meta(
        {
            **dicom_tags,
            "channelsCount": 1,
            "rescaleSlope": 1,
            "rescaleIntercept": 0,
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
        }
    )
    return volume_meta


def inspect_nrrd_series(root_dir: str):
    nrrd_paths = list_files_recursively(root_dir, [".nrrd"])
    logger.info(f"Total {len(nrrd_paths)} nnrd series in directory {root_dir}")
    return nrrd_paths


def read_nrrd_serie_volume(path: str):
    import SimpleITK as sitk
    # find custom NRRD loader in gitlab supervisely_py/-/blob/feature/import-volumes/plugins/import/volumes/src/loaders/nrrd.py
    reader = sitk.ImageFileReader()
    # reader.SetImageIO("NrrdImageIO")
    reader.SetFileName(path)
    sitk_volume = reader.Execute()

    sitk_volume = _sitk_image_orient_ras(sitk_volume)
    f_min_max = sitk.MinimumMaximumImageFilter()
    f_min_max.Execute(sitk_volume)
    meta = get_meta(
        sitk_volume.GetSize(),
        f_min_max.GetMinimum(),
        f_min_max.GetMaximum(),
        sitk_volume.GetSpacing(),
        sitk_volume.GetOrigin(),
        sitk_volume.GetDirection(),
        {},
    )
    return sitk_volume, meta


def read_nrrd_serie_volume_np(paths: List[str]) -> np.ndarray:
    import SimpleITK as sitk
    sitk_volume, meta = read_nrrd_serie_volume(paths)
    volume_np = sitk.GetArrayFromImage(sitk_volume)
    volume_np = np.transpose(volume_np, (2, 1, 0))
    return volume_np, meta
