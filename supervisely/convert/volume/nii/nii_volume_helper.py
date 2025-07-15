import os
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Generator, List, Union

import nrrd
import numpy as np

from supervisely import Api
from supervisely.collection.str_enum import StrEnum
from supervisely.geometry.mask_3d import Mask3D
from supervisely.io.fs import ensure_base_path, get_file_ext, get_file_name
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.volume.volume import convert_3d_nifti_to_nrrd

VOLUME_NAME = "anatomic"
SCORE_NAME = "score"
LABEL_NAME = ["inference", "label", "annotation", "mask", "segmentation"]
MASK_PIXEL_VALUE = "Mask pixel value: "


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


def get_scores_from_table(csv_file_path: str, plane: str) -> dict:
    """Get scores from CSV table and return nested dictionary structure.

    Args:
        csv_file_path: Path to the CSV file containing layer scores

    Returns:
        Nested dictionary with structure:
        {
            "label_index": {
                "slice_index": {
                    "127": {
                        "score": float_value,
                        "comment": ""
                    }
                }
            }
        }
    """
    import csv

    if plane == PlanePrefix.CORONAL:
        plane = "0-1-0"
    elif plane == PlanePrefix.SAGITTAL:
        plane = "1-0-0"
    elif plane == PlanePrefix.AXIAL:
        plane = "0-0-1"

    result = defaultdict(lambda: defaultdict(dict))

    if not os.path.exists(csv_file_path):
        logger.warning(f"CSV file not found: {csv_file_path}")
        return result

    try:
        with open(csv_file_path, "r") as file:
            reader = csv.DictReader(file)
            label_columns = [col for col in reader.fieldnames if col.startswith("Label-")]

            for row in reader:
                frame_idx = int(row["Layer"]) - 1  # Assuming Layer is 1-indexed in CSV

                for label_col in label_columns:
                    label_index = int(label_col.split("-")[1])
                    score = f"{float(row[label_col]):.2f}"
                    result[label_index][plane][frame_idx] = {"score": score, "comment": None}

    except Exception as e:
        logger.warning(f"Failed to read CSV file {csv_file_path}: {e}")
        return {}

    return result


def _find_pixel_values(descr: str) -> int:
    """
    Find the pixel value in the description string.
    """
    lines = descr.split("\n")
    for line in lines:
        if line.strip().startswith(MASK_PIXEL_VALUE):
            try:
                value_part = line.strip().split(MASK_PIXEL_VALUE)[1]
                return int(value_part.strip())
            except (IndexError, ValueError):
                continue
    return None


def get_class_id_to_pixel_value_map(meta: ProjectMeta) -> dict:
    class_id_to_pixel_value = {}
    for obj_class in meta.obj_classes.items():
        pixel_value = _find_pixel_values(obj_class.description)
        if pixel_value is not None:
            class_id_to_pixel_value[obj_class.sly_id] = pixel_value
        elif "Segment_" in obj_class.name:
            try:
                pixel_value = int(obj_class.name.split("_")[-1])
                class_id_to_pixel_value[obj_class.sly_id] = pixel_value
            except (ValueError, IndexError):
                logger.warning(
                    f"Failed to parse pixel value from class name: {obj_class.name}. "
                    "Please ensure the class name ends with a valid integer."
                )
        else:
            logger.warning(
                f"Class {obj_class.name} does not have a pixel value defined in its description. "
                "Please update the class description to include 'Mask pixel value: <value>'."
            )
    return class_id_to_pixel_value


class AnnotationMatcher:
    def __init__(self, items, dataset_id):
        self._ann_paths = defaultdict(list)
        self._item_by_filename = {}
        self._item_by_path = {}

        self.items = items
        self._ds_id = dataset_id

        self._project_wide = False
        self._volumes = None

    @property
    def items(self):
        return self._items

    @items.setter
    def items(self, items):
        self._items = items
        self._ann_paths.clear()
        self._item_by_filename.clear()
        self._item_by_path.clear()

        for item in items:
            path = Path(item.ann_data)
            dataset_name = path.parts[-2]
            filename = path.name

            self._ann_paths[dataset_name].append(filename)
            self._item_by_filename[filename] = item
            self._item_by_path[(dataset_name, filename)] = item

    def get_volumes(self, api: Api):
        dataset_info = api.dataset.get_info_by_id(self._ds_id)
        if dataset_info.items_count > 0:
            datasets = {dataset_info.name: dataset_info}
            self._project_wide = False
        else:
            datasets = {
                dsinfo.name: dsinfo
                for dsinfo in api.dataset.get_list(dataset_info.project_id, recursive=True)
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
        item_to_volume = {}

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
                match = find_best_name_match(ann_name, volume_names)
                if match is not None:
                    if match.plane != ann_name.plane:
                        logger.warning(
                            f"Plane mismatch: {match.plane} != {ann_name.plane} for {ann_file}. Skipping."
                        )
                        continue
                    item_to_volume[self._item_by_filename[ann_file]] = volumes[match.full_name]

        volume_to_items = defaultdict(list)
        for item, volume in item_to_volume.items():
            volume_to_items[volume.id].append(item)

        for volume_id, items in volume_to_items.items():
            if len(items) == 1:
                items[0].is_semantic = True

        # items_to_remove = []
        for item, volume in item_to_volume.items():
            volume_shape = tuple(volume.file_meta["sizes"])
            if item.shape is None:
                continue
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
        type = VOLUME_NAME
    elif SCORE_NAME in full_name and full_name.endswith(".csv"):
        type = SCORE_NAME
    else:
        type = next((part for part in LABEL_NAME if part in full_name), None)
        is_ann = type is not None

    if type is None:
        return

    plane = None
    tokens = name_no_ext.lower().split("_")
    for part in PlanePrefix.values():
        if part in tokens:
            plane = part
            break

    if plane is None:
        return

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


def find_best_name_match(item: NameParts, pool: List[NameParts]) -> Union[NameParts, None]:
    """
    Finds the best matching NameParts object from `pool` for the given annotation NameParts `item`.
    Prefers an exact match where all fields except `type` are the same, and `type` is 'anatomic'.
    Returns the matched NameParts object or None if not found.
    """
    pool_item_names = [i.full_name for i in pool]
    item_name = item.full_name
    # Prefer exact match except for type
    for i in pool:
        if i.name_no_ext == item.name_no_ext.replace(item.type, i.type):
            logger.debug(
                "Found exact match.",
                extra={"item": item_name, "pool_item": i.full_name},
            )
            return i

    logger.debug(
        "Failed to find exact match, trying to find a fallback match UUIDs.",
        extra={"item": item_name, "pool_items": pool_item_names},
    )

    # Fallback: match by plane and patient_uuid, type='anatomic'
    for i in pool:
        if (
            i.plane == item.plane
            and i.patient_uuid == item.patient_uuid
            and i.case_uuid == item.case_uuid
        ):
            logger.debug(
                "Found fallback match for item by UUIDs.",
                extra={"item": item_name, "i": i.full_name},
            )
            return i

    logger.debug(
        "Failed to find fallback match, trying to find a fallback match by plane.",
        extra={"item": item_name, "pool_items": pool_item_names},
    )

    # Fallback: match by plane and type='anatomic'
    for i in pool:
        if i.plane == item.plane:
            logger.debug(
                "Found fallback match for item by plane.",
                extra={"item": item_name, "i": i.full_name},
            )
            return i

    logger.debug(
        "Failed to find any match for item.",
        extra={"item": item_name, "pool_items": pool_item_names},
    )

    return None
