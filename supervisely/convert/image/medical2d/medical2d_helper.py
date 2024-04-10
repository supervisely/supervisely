import os
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import nibabel as nib
except ImportError:
    raise ImportError("No module named nibabel. Please make sure that module is installed from pip and try again.")

import nrrd
import numpy as np
import pydicom
from pydicom import FileDataset

from supervisely import image, logger, volume
from supervisely.io.fs import get_file_ext, get_file_name, get_file_name_with_ext, mkdir


def is_nifti_file(filepath: str) -> bool:
    try:
        nib.load(filepath)
        return True
    except nib.filebasedimages.ImageFileError:
        return False


def convert_nifti_to_nrrd(input_nii_path: str, converted_dir: str) -> Tuple[str, str]:
    nii_img = nib.load(input_nii_path)
    canonical_img = nib.as_closest_canonical(nii_img)
    nii_data = canonical_img.get_fdata()
    affine = canonical_img.affine
    orientation = nib.aff2axcodes(affine)
    nrrd_header = {
        "space": "".join(orientation),
        "space directions": canonical_img.affine[:3, :3].tolist(),
        "sizes": nii_data.shape,
        "type": "float",
        "dimension": len(nii_data.shape),
    }
    output_name = get_file_name(input_nii_path)
    if get_file_ext(output_name) == ".nii":
        output_name = get_file_name(output_name)

    output_nrrd_path = os.path.join(converted_dir, f"nifti_{output_name}.nrrd")
    nrrd.write(output_nrrd_path, nii_data, nrrd_header)

    return output_nrrd_path, output_name


def is_dicom_file(path: str) -> bool:
    """Checks if file is dicom file by given path."""
    try:
        pydicom.read_file(str(Path(path).resolve()), stop_before_pixels=True)
        result = True
    except Exception as ex:
        logger.warn("'{}' appears not to be a DICOM file\n({})".format(path, ex))
        result = False
    return result


def check_nrrd(path):
    try:
        img = nrrd.read(path)
        return True
    except:
        return False


def create_nrrd_header_from_dcm(dcm_path: str, frame_axis: int = 2) -> dict:
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


def convert_dcm_to_nrrd(image_path: str, converted_dir: str) -> Tuple[List[str], List[str]]:
    """Converts DICOM data to nrrd format and returns image paths and image names"""
    original_name = get_file_name_with_ext(image_path)
    curr_convert_dir = os.path.join(converted_dir, original_name)
    mkdir(curr_convert_dir)

    dcm = pydicom.read_file(image_path)
    pixel_data_list = [dcm.pixel_array]
    if len(dcm.pixel_array.shape) == 3:
        if dcm.pixel_array.shape[0] == 1 and not hasattr(dcm, "NumberOfFrames"):
            frames = 1
            pixel_data_list = [
                dcm.pixel_array.reshape((dcm.pixel_array.shape[1], dcm.pixel_array.shape[2]))
            ]
            header = create_nrrd_header_from_dcm(image_path)
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
        header = create_nrrd_header_from_dcm(image_path)
    else:
        raise NotImplementedError(
            f"this type of dcm data is not supported, pixel_array.shape = {len(dcm.pixel_array.shape)}"
        )

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
    return save_paths, image_names
