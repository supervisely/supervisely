import os
import re
from os.path import basename, dirname, exists, join, normpath, pardir
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nrrd
import numpy as np
import pydicom
from pydicom import FileDataset
from tqdm import tqdm

from supervisely import image, logger, volume
from supervisely.annotation.tag import Tag
from supervisely.io.fs import (
    dir_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    mkdir,
)

_MEDICAL_DEFAULT_GROUP_TAG_NAMES = [
    "StudyInstanceUID",
    "StudyID",
    "SeriesInstanceUID",
    "TreatmentSessionUID",
    "Manufacturer",
    "ManufacturerModelName",
    "Modality",
]


def slice_nifti_file(nii_file_path: str, converted_dir: str) -> Tuple[List[str], List[str]]:
    """Slices file into 2D slices if it is 3D.
    Returns image paths and image names.

    This function always slices 3D nifti files into 2D slices by axial orientation because source data automatically gets converted to RAS coordinate system.
    """

    output_name = get_file_name(nii_file_path)
    if get_file_ext(output_name) == ".nii":
        output_name = get_file_name(output_name)

    data, header = volume.convert_nifti_to_nrrd(nii_file_path)

    output_paths = []
    output_names = []
    if len(data.shape) >= 3:
        indices = [slice(None)] * 3 + [0] * (data.ndim - 3)

        for i in range(data.shape[2]):
            indices[2] = i
            slice_data = data[tuple(indices)]

            new_header = {
                "sizes": slice_data.shape,
                "type": header.get("type", "float"),
                "dimension": len(slice_data.shape),
            }
            output_path = os.path.join(converted_dir, f"nifti_{output_name}_{i}.nrrd")
            output_names.append(get_file_name_with_ext(output_path))
            nrrd.write(output_path, slice_data, new_header)
            output_paths.append(output_path)
    else:
        output_path = os.path.join(converted_dir, f"nifti_{output_name}.nrrd")
        output_names.append(get_file_name_with_ext(output_path))
        nrrd.write(output_path, data, header)
        output_paths.append(output_path)

    return output_paths, output_names


def is_dicom_file(path: str) -> bool:
    """Checks if file is dicom file by given path."""
    try:
        pydicom.read_file(str(Path(path).resolve()), stop_before_pixels=True)
        result = True
    except Exception as ex:
        logger.warning("'{}' appears not to be a DICOM file\n({})".format(path, ex))
        result = False
    return result


def check_nrrd(path):
    try:
        nrrd.read(path)
        return True
    except:
        return False


def create_nrrd_header_from_dcm(dcm_path: str, frame_axis: int = 2) -> dict:
    """
    Create nrrd header from DICOM file.

    Assume that axes will be in the RAS order (right-anterior-superior).
    """
    _, meta = volume.read_dicom_serie_volume([dcm_path], False)
    dimensions: Dict = meta.get("dimensionsIJK")
    header = {
        "type": "float",
        "sizes": [dimensions.get("x"), dimensions.get("y")],
        "dimension": 2,
        "space": "right-anterior-superior",
    }

    if frame_axis == 0:
        spacing = meta["spacing"][1:]
        header["space directions"] = [[spacing[0], 0], [0, spacing[1]]]
    if frame_axis == 1:
        spacing = meta["spacing"][0::2]
        header["space directions"] = [[spacing[0], 0], [0, spacing[1]]]
    if frame_axis == 2:
        spacing = meta["spacing"][0:2]
        header["space directions"] = [[spacing[0], 0], [0, spacing[1]]]
    return header


def find_frame_axis(pixel_data: np.ndarray, frames: int) -> int:
    for axis in range(len(pixel_data.shape)):
        if pixel_data.shape[axis] == frames:
            return axis
    raise ValueError("Unable to recognize the frame axis for splitting a set of images")


def create_pixel_data_set(dcm: FileDataset, frame_axis: int) -> Tuple[List[np.ndarray], int]:
    if frame_axis == 0:
        pixel_array = np.transpose(dcm.pixel_array, (2, 1, 0))
    elif frame_axis == 1:
        pixel_array = np.transpose(dcm.pixel_array, (2, 0, 1))
    else:
        pixel_array = dcm.pixel_array
    frame_axis = 2
    list_of_images = np.split(pixel_array, int(dcm.NumberOfFrames), axis=frame_axis)
    return list_of_images, frame_axis

def convert_to_monochrome2(dcm_path: str, dcm: FileDataset) -> FileDataset:
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

        logger.info("Rewriting DICOM file with monochrome2 format")
        dcm.save_as(dcm_path)
    return dcm

def convert_dcm_to_nrrd(
    image_path: str, converted_dir: str, group_tag_name: Optional[list] = None
) -> Tuple[List[str], List[str], List[dict], dict]:
    """Converts DICOM data to nrrd format and returns image paths and image names"""
    original_name = get_file_name_with_ext(image_path)
    if not dir_exists(converted_dir):
        mkdir(converted_dir)
    curr_convert_dir = os.path.join(converted_dir, original_name)
    mkdir(curr_convert_dir)

    dcm = pydicom.read_file(image_path)
    try:
        if dcm.file_meta.TransferSyntaxUID.is_compressed:
            dcm.decompress()
    except AttributeError:
        logger.warn("Couldn't find key 'TransferSyntaxUID' in dicom's metadata.")
    dcm = convert_to_monochrome2(image_path, dcm)

    dcm_meta = get_dcm_meta(dcm)

    if group_tag_name is None:
        group_tag_names = _MEDICAL_DEFAULT_GROUP_TAG_NAMES
    else:
        group_tag_names = [group_tag_name] + _MEDICAL_DEFAULT_GROUP_TAG_NAMES

    is_group_tag_exists = next((name for name in group_tag_names if name in dcm), None) is not None
    if not is_group_tag_exists:
        logger.warning(f"Not found group tags in DICOM file.")

    header, frames, frame_axis, pixel_data_list = get_nrrd_data(image_path, dcm)

    group_tags = []
    save_paths = []
    image_names = []
    frames_list = [f"{i:0{len(str(frames))}d}" for i in range(1, frames + 1)]

    for pixel_data, frame_number in zip(pixel_data_list, frames_list):
        if frames == 1:
            pixel_data = image.rotate(img=pixel_data, degrees_angle=270)
            pixel_data = image.fliplr(pixel_data)
            image_name = f"{original_name}.nrrd"
        else:
            pixel_data = np.squeeze(pixel_data, frame_axis)
            image_name = f"{frame_number}_{original_name}.nrrd"

        save_path = os.path.join(curr_convert_dir, image_name)
        nrrd.write(save_path, pixel_data, header)
        save_paths.append(save_path)
        image_names.append(image_name)

    for tag_name in group_tag_names:
        if dcm.get(tag_name) is not None:
            group_tag_value = dcm[tag_name].value
            if group_tag_value is not None and group_tag_value != "":
                group_tags.append({"name": tag_name, "value": group_tag_value})
        elif group_tag_name is not None and tag_name == group_tag_name:
            logger.warning(f"Couldn't find key {group_tag_name!r} in file's metadata.")

    return save_paths, image_names, group_tags, dcm_meta


def slice_nrrd_file(nrrd_file_path: str, output_dir: str) -> Tuple[List[str], List[str]]:
    """Slices nrrd file into 2D slices if it is 3D. Returns image paths and image names.

    This function always slices 3D nrrd files into 2D slices by axial orientation because source data automatically gets converted to RAS coordinate system.
    Data with dimensions more than 3 will be truncated to 3D.
    """
    data, header = nrrd.read(nrrd_file_path)
    output_paths = []
    output_names = []

    if header.get("dimension", 0) > 2:
        logger.info(f"File [{nrrd_file_path}] have more than 2 dimensions.")
        try:
            domain_count = 0
            if "kinds" in header:
                kinds = header.get("kinds")
                domain_indices = [i for i, kind in enumerate(kinds) if kind == "domain"]
                domain_count = len(domain_indices)
            else:
                domain_indices = [0, 1, 2]
                domain_count = 3

            # If there are more than three domain dimensions, ignore the extra ones
            domain_indices = domain_indices[:3]

            # Create a new array with only the first three domain dimensions
            indices = [0] * data.ndim
            for idx in domain_indices:
                indices[idx] = slice(None)
            data = data[tuple(indices)]
            domain_indices = [0, 1, 2]

            # If shape is 3D and space is not RAS, then we need to transpose the data
            if domain_count == 3 and "space" in header:
                if header.get("space").lower() not in ["right-anterior-superior", "ras"]:
                    domain_indices = domain_indices[::-1]
                    data = np.transpose(data, domain_indices)

            progress = tqdm(range(data.shape[-1]), desc="Slicing images")

            for i in range(data.shape[-1]):
                slice_data = np.take(data, i, axis=len(data.shape) - 1)

                new_header = {
                    "sizes": slice_data.shape,
                    "type": header.get("type", "float"),
                    "dimension": len(slice_data.shape),
                }

                output_nrrd_path = os.path.join(
                    output_dir, f"{os.path.basename(nrrd_file_path).replace('.nrrd', '')}_{i}.nrrd"
                )
                output_name = get_file_name_with_ext(output_nrrd_path)
                nrrd.write(output_nrrd_path, slice_data, new_header)
                output_paths.append(output_nrrd_path)
                output_names.append(output_name)
                progress.update(1)
            progress.close()
        except Exception:
            logger.warn(f"File [{nrrd_file_path}] is not supported. Skipping...")
    else:
        output_paths.append(nrrd_file_path)
        output_names.append(get_file_name_with_ext(nrrd_file_path))
    return output_paths, output_names


def get_dcm_meta(dcm: FileDataset) -> List[Tag]:
    """Create tags from DICOM metadata."""
    from supervisely.volume.volume import _anonymize_tags

    filtered_tags = []

    filename = get_file_name_with_ext(dcm.filename)
    empty_tags, too_long_tags = [], []
    for dcm_tag in dcm.keys():
        try:
            curr_tag = dcm[dcm_tag]
            dcm_tag_name = str(curr_tag.name)
            dcm_tag_value = str(curr_tag.value)
            if dcm_tag_name in ["Patient's Name", "Patient ID"] + _anonymize_tags:
                dcm_tag_value = "anonymized"
            if dcm_tag_value in ["", None]:
                empty_tags.append(dcm_tag_name)
                continue
            if len(dcm_tag_value) > 255:
                too_long_tags.append(dcm_tag_name)
                continue
            filtered_tags.append((dcm_tag_name, dcm_tag_value))
        except KeyError:
            dcm_filename = get_file_name_with_ext(dcm.filename)
            logger.warning(f"Couldn't find key '{dcm_tag}' in file's metadata: '{dcm_filename}'")
            continue

    if len(empty_tags) > 0:
        logger.warning(f"{filename}: {len(dcm_tag_name)} tags have empty value. Skipped tags: {empty_tags}.")
    if len(too_long_tags) > 0:
        logger.warning(
            f"{filename}: {len(too_long_tags)} tags have too long value (> 255 symbols). Skipped tags: {too_long_tags}."
        )

    dcm_tags_dict = {}
    for dcm_tag_name, dcm_tag_value in filtered_tags:
        dcm_tags_dict[dcm_tag_name] = dcm_tag_value
    return dcm_tags_dict


def get_nrrd_data(image_path: str, dcm: FileDataset):

    frame_axis = 2
    pixel_data_list = [dcm.pixel_array]

    if len(dcm.pixel_array.shape) == 3:
        if dcm.pixel_array.shape[0] == 1 and not hasattr(dcm, "NumberOfFrames"):
            frames = 1
            pixel_data_list = [
                dcm.pixel_array.reshape((dcm.pixel_array.shape[1], dcm.pixel_array.shape[2]))
            ]
            header = create_nrrd_header_from_dcm(image_path, frame_axis)
        else:
            try:
                frames = int(dcm.NumberOfFrames)
            except AttributeError as e:
                if str(e) == "'FileDataset' object has no attribute 'NumberOfFrames'":
                    e.args = ("can't get 'NumberOfFrames' from dcm meta.",)
                    raise e
            frame_axis = find_frame_axis(dcm.pixel_array, frames)
            pixel_data_list, frame_axis = create_pixel_data_set(dcm, frame_axis)
            header = create_nrrd_header_from_dcm(image_path, frame_axis)
    elif len(dcm.pixel_array.shape) == 2:
        frames = 1
        header = create_nrrd_header_from_dcm(image_path, frame_axis)
    else:
        raise RuntimeError(
            f"This type of dcm data is not supported, pixel_array.shape = {len(dcm.pixel_array.shape)}"
        )

    return header, frames, frame_axis, pixel_data_list
