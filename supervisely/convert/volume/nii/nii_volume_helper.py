import os
from typing import Generator

import nrrd
import numpy as np
from pathlib import Path
from collections import defaultdict, namedtuple

from supervisely import Api
from supervisely.collection.str_enum import StrEnum
from supervisely.geometry.mask_3d import Mask3D
from supervisely.io.fs import ensure_base_path, get_file_ext, get_file_name
from supervisely.sly_logger import logger
from supervisely.volume.volume import convert_3d_nifti_to_nrrd

VOLUME_NAME = "anatomic"
LABEL_NAME = ["inference", "label", "annotation", "mask", "segmentation"]


class PlanePrefix(str, StrEnum):
    """Prefix for plane names."""

    CORONAL = "cor"
    SAGITTAL = "sag"
    AXIAL = "axl"


def read_cls_color_map(path: str) -> dict:
    """Read class color map from TXT file.

    ```txt
    1 Femur 255 0 0
    2 Femoral cartilage 0 255 0
    3 Tibia 0 0 255
    4 Tibia cartilage 255 255 0
    5 Patella 0 255 255
    6 Patellar cartilage 255 0 255
    7 Miniscus 175 175 175
    ```
    """

    cls_color_map = {}
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as file:
            for line in file:
                parts = line.strip().split()
                color = None
                try:
                    color = list(map(int, parts[-3:]))
                except:
                    pass
                cls_id = int(parts[0])
                if not color:
                    cls_name = " ".join(parts[1:])
                else:
                    cls_name = " ".join(parts[1:-3])
                if cls_id in cls_color_map:
                    logger.warning(f"Duplicate class ID {cls_id} found in color map.")
                if cls_name in cls_color_map:
                    logger.warning(f"Duplicate class name {cls_name} found in color map.")
                if len(color) != 3:
                    logger.warning(f"Invalid color format for class {cls_name}. Expected 3 values.")
                if any(c < 0 or c > 255 for c in color):
                    logger.warning(
                        f"Invalid color value for class {cls_name}. Expected values between 0 and 255."
                    )
                cls_color_map[cls_id] = (cls_name, color)
    except Exception as e:
        logger.warning(f"Failed to read class color map from {path}: {e}")
        return None
    return cls_color_map

def read_json_map(path: str) -> dict:
    import json

    """Read JSON map from file."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as file:
            json_map = json.load(file)
    except Exception as e:
        logger.warning(f"Failed to read JSON map from {path}: {e}")
        return None
    return json_map


def nifti_to_nrrd(nii_file_path: str, converted_dir: str) -> str:
    """Convert NIfTI 3D volume file to NRRD 3D volume file."""

    output_name = get_file_name(nii_file_path)
    if get_file_ext(output_name) == ".nii":
        output_name = get_file_name(output_name)

    data, header = convert_3d_nifti_to_nrrd(nii_file_path)

    nrrd_file_path = os.path.join(converted_dir, f"{output_name}.nrrd")
    ensure_base_path(nrrd_file_path)

    nrrd.write(nrrd_file_path, data, header)
    return nrrd_file_path


def get_annotation_from_nii(path: str) -> Generator[Mask3D, None, None]:
    """Get annotation from NIfTI 3D volume file."""

    data, _ = convert_3d_nifti_to_nrrd(path)
    unique_classes = np.unique(data)

    for class_id in unique_classes:
        if class_id == 0:
            continue
        mask = Mask3D(data == class_id)
        yield mask, class_id

class AnnotationMatcher:
    def __init__(self, items, dataset_id):
        self._items = items
        self._ds_id = dataset_id
        self._ann_paths = defaultdict(list)

        self._item_by_filename = {}
        self._item_by_path = {}

        for item in items:
            path = Path(item.ann_data)
            dataset_name = path.parts[-2]
            filename = path.name

            self._ann_paths[dataset_name].append(filename)
            self._item_by_filename[filename] = item
            self._item_by_path[(dataset_name, filename)] = item

        self._project_wide = False
        self._volumes = None

    def get_volumes(self, api: Api):
        if self._ds_id is None:
            raise ValueError("Dataset ID is not set.")
        dataset_info = api.dataset.get_info_by_id(self._ds_id)
        project_id = dataset_info.project_id
        if len(self._ann_paths.keys()) == 1:
            self._project_wide = False
            self._volumes = {get_file_name(info.name): info for info in api.volume.get_list(self._ds_id)}
            return

        datasets = {dsinfo.name: dsinfo for dsinfo in api.dataset.get_list(project_id, recursive=True)}
        volumes = defaultdict(lambda: {})
        for ds_name, ds_info in datasets.items():
            if ds_name in self._ann_paths:
                volumes[ds_name].update(
                    {info.name: info for info in api.volume.get_list(ds_info.id)}
                )

        if len(volumes) == 0:
            err_msg = None
            if not self._project_wide:
                err_msg = f"Failed to retrieve volumes from the dataset {self._ds_id}."
            else:
                err_msg = "Failed to retrieve volumes from the project. Perhaps the input data structure is incorrect."
            raise RuntimeError(err_msg)
        self._project_wide = True if len(volumes) > 1 else False
        self._volumes = volumes

    def match_items(self):
        """Match annotation files with corresponding volumes."""
        def to_volume_name(name):
            if name.endswith(".nii.gz"):
                name = name.replace(".nii.gz", ".nrrd")
            elif name.endswith(".nii"):
                name = name.replace(".nii", ".nrrd") 
            if "_" not in name:
                return None
            name_parts = get_file_name(name).split("_")[:3]
            return f"{name_parts[0]}_{VOLUME_NAME}_{name_parts[2]}"

        item_to_volume = {}

        if self._project_wide:
            # Project-wide matching
            for dataset_name, volumes in self._volumes.items():
                volumes_copy = volumes.copy()
                for ann_file in self._ann_paths[dataset_name]:
                    expected_volume_name = to_volume_name(ann_file)
                    if expected_volume_name is None:
                        logger.warning(f"Invalid volume name for {ann_file}. Skipping.")
                        continue
                    if expected_volume_name in volumes_copy:
                        volume = volumes_copy[expected_volume_name]
                        item = self._item_by_path.get((dataset_name, ann_file))
                        if item:
                            item_to_volume[item] = volume
                            # Remove the volume from the pool after matching
                            del volumes_copy[expected_volume_name]
                        else:
                            logger.warning(f"Item not found for {ann_file} in dataset {dataset_name}.")
                    else:
                        logger.warning(
                            f"Volume name {expected_volume_name} not found in dataset {dataset_name}."
                        )
        else:
            # Dataset-wide matching
            dataset_name = list(self._ann_paths.keys())[0]
            volumes_copy = self._volumes.copy()
            for ann_file in self._ann_paths[dataset_name]:
                expected_volume_name = to_volume_name(ann_file)
                if expected_volume_name in volumes_copy:
                    item = self._item_by_filename.get(ann_file)
                    if item:
                        item_to_volume[item] = volumes_copy[expected_volume_name]
                        # Remove the volume from the pool after matching
                        del volumes_copy[expected_volume_name]
                    else:
                        logger.warning(f"Item not found for {ann_file} in single dataset mode.")
                else:
                    logger.warning(
                        f"Volume name {expected_volume_name} not found in dataset {self._ds_id}."
                    )

        # validate shape
        for item, volume in item_to_volume.items():
            if item.shape != item.volume_meta.shape:
                logger.warning(
                    f"Volume shape mismatch: {item.shape} != {item.volume_meta.shape}. Skipping item."
                )
                del item_to_volume[item]

        return item_to_volume

    def match_from_json(self, api: Api, json_map: dict):
        """
        Match annotation files with corresponding volumes based on a JSON map.

        Example json structure:
        {
            "cor_inference_1.nii": 123,
            "sag_mask_2.nii": 456
        }
        Where key is the annotation file name and value is the volume ID.
        """
        item_to_volume = {}
        for ann_name, volume_id in json_map.items():
            item = self._item_by_filename.get(ann_name)
            if item:
                volume = api.volume.get_info_by_id(volume_id)
                if volume:
                    item_to_volume[item] = volume
                else:
                    logger.warning(f"Volume {volume_id} not found in project.")
            else:
                logger.warning(f"Item not found for annotation file {ann_name}.")
        return item_to_volume
