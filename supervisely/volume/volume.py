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

    meta["windowWidth"] = int(meta["windowWidth"])
    meta["windowCenter"] = int(meta["windowCenter"])
    meta["rescaleSlope"] = int(meta["rescaleSlope"])
    meta["rescaleIntercept"] = int(meta["rescaleIntercept"])

    return meta


def read_serie_volume_np(paths: List[str]) -> np.ndarray:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(paths)
    volume = reader.Execute()
    volume = sitk.DICOMOrient(volume, "RAS")
    volume_np = sitk.GetArrayFromImage(volume)
    return volume_np


def read_dicom_tags(path, allowed_keys: Union[None, List[str]] = None):
    # allowed_keys = [
    #     "SeriesInstanceUID",
    #     "Modality",
    #     "PatientID",
    #     "PatientName",
    #     "WindowCenter",
    #     "WindowWidth",
    #     "RescaleIntercept",
    #     "RescaleSlope",
    #     "PhotometricInterpretation",
    # ]
    photometricInterpretationRGB = set(
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
            vol_info[keyword] = float(vol_info[keyword])
        elif (
            keyword == "photometricInterpretation" and v in photometricInterpretationRGB
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
    from supervisely.volume.characterize_data import inspect_series as _inspect_series

    series_pd = _inspect_series(root_dir)
    series_infos = json.loads(series_pd.to_json(orient="index"))

    res = []
    for index_str, info in series_infos.items():
        files = info["files"]
        tags = read_dicom_tags(files[0])

        shape = info["image size"]
        # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/01_Image_Basics.html#Accessing-Attributes
        width = shape[0]
        height = shape[1]
        depth = shape[2]

        min_intensity = info["min intensity"]
        max_intensity = info["max intensity"]

        spacing = info["image spacing"]
        origin = info["image origin"]
        directions = info["axis direction"]

        meta = get_meta(
            [width, height, depth],
            min_intensity,
            max_intensity,
            spacing,
            origin,
            directions,
            tags,
        )

        res.append((files, meta))
    return res


def get_meta(
    np_shape, min_intensity, max_intensity, spacing, origin, directions, dicom_tags={}
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
                "x": np_shape[0],
                "y": np_shape[1],
                "z": np_shape[2],
            },
            "ACS": "RAS",
            # instead of IJK2WorldMatrix field
            "spacing": spacing,
            "origin": origin,
            "directions": directions,
        }
    )
    return volume_meta
