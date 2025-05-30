import os
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Generator

import nrrd
import numpy as np

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

    data, header = convert_3d_nifti_to_nrrd(path)
    unique_classes = np.unique(data)

    for class_id in unique_classes:
        if class_id == 0:
            continue
        mask = Mask3D(data == class_id, volume_header=header)
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
        dataset_info = api.dataset.get_info_by_id(self._ds_id)
        datasets = {dataset_info.name: dataset_info}
        project_id = dataset_info.project_id
        if dataset_info.items_count > 0 and len(self._ann_paths.keys()) == 1:
            self._project_wide = False
        else:
            datasets = {
                dsinfo.name: dsinfo for dsinfo in api.dataset.get_list(project_id, recursive=True)
            }
            self._project_wide = True

        volumes = defaultdict(lambda: {})
        ds_filter = lambda ds_name: ds_name in self._ann_paths if self._project_wide else True
        for ds_name, ds_info in datasets.items():
            if ds_filter(ds_name):
                volumes[ds_name].update(
                    {info.name: info for info in api.volume.get_list(ds_info.id)}
                )

        if len(volumes) == 0:
            err_msg = "Failed to retrieve volumes from the project. Perhaps the input data structure is incorrect."
            raise RuntimeError(err_msg)

        self._volumes = volumes

    def match_items(self):
        """Match annotation files with corresponding volumes using regex-based matching."""
        import re

        item_to_volume = {}

        # Perform matching
        for dataset_name, volumes in self._volumes.items():
            volume_names = [parse_name_parts(name) for name in list(volumes.keys())]
            _volume_names = [vol for vol in volume_names if vol is not None]
            if len(_volume_names) == 0:
                logger.warning(f"No valid volume names found in dataset {dataset_name}.")
                continue
            elif len(_volume_names) != len(volume_names):
                logger.debug(f"Some volume names in dataset {dataset_name} could not be parsed.")
            volume_names = _volume_names

            ann_files = (
                self._ann_paths.get(dataset_name, [])
                if self._project_wide
                else list(self._ann_paths.values())[0]
            )
            for ann_file in ann_files:
                ann_name = parse_name_parts(ann_file)
                if ann_name is None:
                    logger.warning(f"Failed to parse annotation name: {ann_file}")
                    continue
                match = find_best_volume_match_for_ann(ann_name, volume_names)
                if match is not None:
                    if match.plane != ann_name.plane:
                        logger.warning(
                            f"Plane mismatch: {match.plane} != {ann_name.plane} for {ann_file}. Skipping."
                        )
                        continue
                    item_to_volume[self._item_by_filename[ann_file]] = volumes[match.full_name]

        # Mark volumes having only one matching item as semantic and validate shape.
        volume_to_items = defaultdict(list)
        for item, volume in item_to_volume.items():
            volume_to_items[volume.id].append(item)

        for volume_id, items in volume_to_items.items():
            if len(items) == 1:
                items[0].is_semantic = True

        # items_to_remove = []
        for item, volume in item_to_volume.items():
            volume_shape = tuple(volume.file_meta["sizes"])
            if item.shape != volume_shape:
                logger.warning(f"Volume shape mismatch: {item.shape} != {volume_shape}")
                # items_to_remove.append(item)
        # for item in items_to_remove:
        # del item_to_volume[item]

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


NameParts = namedtuple(
    "NameParts",
    [
        "full_name",
        "name_no_ext",
        "type",
        "plane",
        "is_ann",
        "patient_uuid",
        "case_uuid",
        "ending_idx",
    ],
)


def parse_name_parts(full_name: str) -> NameParts:
    from uuid import UUID

    name = get_file_name(full_name)
    if name.endswith(".nii"):
        name = get_file_name(name)
    name_no_ext = name

    type = None
    is_ann = False
    if VOLUME_NAME in full_name:
        type = "anatomic"
    else:
        type = next((part for part in LABEL_NAME if part in full_name), None)
        is_ann = type is not None

    if type is None:
        return

    plane = None
    for part in PlanePrefix.values():
        if part in name:
            plane = part
            break

    if plane is None:
        return

    is_ann = any(part in name.lower() for part in LABEL_NAME)

    patient_uuid = None
    case_uuid = None

    if len(name_no_ext) > 73:
        try:
            uuids = name_no_ext[:73].split("_")
            if len(uuids) != 2:
                raise ValueError("Invalid UUID format")
            patient_uuid = UUID(name_no_ext[:36])
            case_uuid = UUID(name_no_ext[37:73])
        except ValueError:
            logger.debug(
                f"Failed to parse UUIDs from name: {name_no_ext}.",
                extra={"full_name": full_name},
            )
            patient_uuid = None
            case_uuid = None

    try:
        ending_idx = name_no_ext.split("_")[-1]
        if ending_idx.isdigit():
            ending_idx = int(ending_idx)
        else:
            ending_idx = None
    except ValueError:
        ending_idx = None
        logger.debug(
            f"Failed to parse ending index from name: {name_no_ext}.",
            extra={"full_name": full_name},
        )

    return NameParts(
        full_name=full_name,
        name_no_ext=name_no_ext,
        type=type,
        plane=plane,
        is_ann=is_ann,
        patient_uuid=patient_uuid,
        case_uuid=case_uuid,
        ending_idx=ending_idx,
    )


def find_best_volume_match_for_ann(ann, volumes):
    """
    Finds the best matching NameParts object from `volumes` for the given annotation NameParts `ann`.
    Prefers an exact match where all fields except `type` are the same, and `type` is 'anatomic'.
    Returns the matched NameParts object or None if not found.
    """
    volume_names = [volume.full_name for volume in volumes]
    ann_name = ann.full_name
    # Prefer exact match except for type
    for vol in volumes:
        if vol.name_no_ext == ann.name_no_ext.replace(ann.type, "anatomic"):
            logger.debug(
                "Found exact match for annotation.",
                extra={"ann": ann_name, "vol": vol.full_name},
            )
            return vol

    logger.debug(
        "Failed to find exact match, trying to find a fallback match UUIDs.",
        extra={"ann": ann_name, "volumes": volume_names},
    )

    # Fallback: match by plane and patient_uuid, type='anatomic'
    for vol in volumes:
        if (
            vol.plane == ann.plane
            and vol.patient_uuid == ann.patient_uuid
            and vol.case_uuid == ann.case_uuid
        ):
            logger.debug(
                "Found fallback match for annotation by UUIDs.",
                extra={"ann": ann_name, "vol": vol.full_name},
            )
            return vol

    logger.debug(
        "Failed to find fallback match, trying to find a fallback match by plane.",
        extra={"ann": ann_name, "volumes": volume_names},
    )

    # Fallback: match by plane and type='anatomic'
    for vol in volumes:
        if vol.plane == ann.plane:
            logger.debug(
                "Found fallback match for annotation by plane.",
                extra={"ann": ann_name, "vol": vol.full_name},
            )
            return vol

    logger.debug(
        "Failed to find any match for annotation.",
        extra={"ann": ann_name, "volumes": volume_names},
    )

    return None
