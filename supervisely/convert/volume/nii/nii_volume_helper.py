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
            if len(self._volumes) == 0:
                raise RuntimeError(f"No volumes found in the dataset (id: {self._ds_id}).")
            return

        datasets = {dsinfo.name: dsinfo for dsinfo in api.dataset.get_list(project_id, recursive=True)}
        volumes = defaultdict(lambda: {})
        for ds_name, ds_info in datasets.items():
            if ds_name in self._ann_paths:
                volumes[ds_name].update(
                    {info.name: info for info in api.volume.get_list(ds_info.id)}
                )
                self._project_wide = True

        if len(volumes) == 0:
            err_msg = None
            if not self._project_wide:
                err_msg = f"Failed to retrieve volumes from the dataset {self._ds_id}."
            else:
                err_msg = "Failed to retrieve volumes from the project. Perhaps the input data structure is incorrect."
            raise RuntimeError(err_msg)

        self._volumes = volumes

    def match_items(self):
        """Match annotation files with corresponding volumes using regex-based matching."""
        import re

        def extract_prefix(ann_file):
            import re
            pattern = r'^(?P<prefix>cor|sag|axl).*?(?:' + "|".join(LABEL_NAME) + r').*\.nii(?:\.gz)?$'
            m = re.match(pattern, ann_file, re.IGNORECASE)
            if m:
                return m.group("prefix").lower()
            return None

        def is_volume_match(volume_name, prefix):
            pattern = r'^' + re.escape(prefix) + r'.*?anatomic.*\.nii(?:\.gz)?$'
            return re.match(pattern, volume_name, re.IGNORECASE) is not None

        def find_best_volume_match(prefix, available_volumes):
            candidates = {name: vol for name, vol in available_volumes.items() if is_volume_match(name, prefix)}
            if not candidates:
                return None, None

            # Prefer an exact candidate
            exact_candidate = re.sub(r'(' + '|'.join(LABEL_NAME) + r')', 'anatomic', ann_file, flags=re.IGNORECASE)
            for name in candidates:
                if re.fullmatch(re.escape(exact_candidate) + r'(\.nrrd)', name, re.IGNORECASE):
                    return name, candidates[name]

            # Otherwise, choose the candidate with the shortest name
            best_match = sorted(candidates.keys(), key=len)[0]
            return best_match, candidates[best_match]

        item_to_volume = {}

        def process_annotation_file(ann_file, dataset_name, volumes):
            prefix = extract_prefix(ann_file)
            if prefix is None:
                logger.warning(f"Failed to extract prefix from annotation file {ann_file}. Skipping.")
                return

            matched_name, matched_volume = find_best_volume_match(prefix, volumes)
            if not matched_volume:
                logger.warning(f"No matching volume found for annotation with prefix '{prefix}' in dataset {dataset_name}.")
                return

            # Retrieve the correct item based on matching mode.
            item = (
                self._item_by_path.get((dataset_name, ann_file))
                if self._project_wide
                else self._item_by_filename.get(ann_file)
            )
            if not item:
                logger.warning(f"Item not found for annotation file {ann_file} in {'dataset ' + dataset_name if self._project_wide else 'single dataset mode'}.")
                return

            item_to_volume[item] = matched_volume
            if matched_name.lower() != f"{prefix}_anatomic".lower():
                logger.debug(f"Fuzzy matched {ann_file} to volume {matched_name} using prefix '{prefix}'.")

        # Perform matching for project-wide or dataset-wide scenarios.
        if self._project_wide:
            for dataset_name, volumes in self._volumes.items():
                for ann_file in self._ann_paths[dataset_name]:
                    process_annotation_file(ann_file, dataset_name, volumes)
        else:
            dataset_name = list(self._ann_paths.keys())[0]
            for ann_file in self._ann_paths[dataset_name]:
                process_annotation_file(ann_file, dataset_name, self._volumes)

        # Mark volumes having only one matching item as semantic and validate shape.
        volume_to_items = defaultdict(list)
        for item, volume in item_to_volume.items():
            volume_to_items[volume.id].append(item)
        
        for volume_id, items in volume_to_items.items():
            if len(items) == 1:
                items[0].is_semantic = True

        items_to_remove = []
        for item, volume in item_to_volume.items():
            volume_shape = tuple(volume.file_meta["sizes"])
            if item.shape != volume_shape:
                logger.warning(f"Volume shape mismatch: {item.shape} != {volume_shape}. Skipping item.")
                items_to_remove.append(item)
        for item in items_to_remove:
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

        For project-wide matching, the key should include dataset name:
        {
            "dataset1/cor_inference_1.nii": 123,
            "dataset2/sag_mask_2.nii": 456
        }
        """
        item_to_volume = {}

        for ann_path, volume_id in json_map.items():
            # Check if it's a project-wide path (contains dataset name)
            path_parts = Path(ann_path)
            if len(path_parts.parts) > 1:
                # Project-wide format: "dataset_name/filename.nii"
                dataset_name = path_parts.parts[-2]
                ann_name = path_parts.name
                item = self._item_by_path.get((dataset_name, ann_name))
            else:
                # Single dataset format: "filename.nii"
                ann_name = path_parts.name
                item = self._item_by_filename.get(ann_name)

            if item:
                volume = api.volume.get_info_by_id(volume_id)
                if volume:
                    item_to_volume[item] = volume

                    # Validate shape
                    volume_shape = tuple(volume.file_meta["sizes"])
                    if item.shape != volume_shape:
                        logger.warning(
                            f"Volume shape mismatch: {item.shape} != {volume_shape} for {ann_path}. Using anyway."
                        )
                else:
                    logger.warning(f"Volume {volume_id} not found for {ann_path}.")
            else:
                logger.warning(f"Item not found for annotation file {ann_path}.")

        # Set semantic flag for volumes with only one associated item
        volume_to_items = defaultdict(list)
        for item, volume in item_to_volume.items():
            volume_to_items[volume.id].append(item)
        for volume_id, items in volume_to_items.items():
            if len(items) == 1:
                items[0].is_semantic = True

        return item_to_volume
