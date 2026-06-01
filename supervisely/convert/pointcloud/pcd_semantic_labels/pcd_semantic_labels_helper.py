import struct
from collections import defaultdict
from typing import Dict, List, Optional

from supervisely import logger
from supervisely.io.json import load_json_file


def read_class_mapping(mapping_path: str) -> Dict[int, str]:
    data = load_json_file(mapping_path)
    if not isinstance(data, dict):
        raise ValueError("class_mapping.json must contain an object with label IDs as keys.")

    mapping = {}
    for raw_label_id, raw_class in data.items():
        if not isinstance(raw_class, str) or raw_class == "":
            raise ValueError(f"Class name for label ID {raw_label_id!r} is empty or invalid.")
        for label_id in _parse_mapping_key(raw_label_id):
            if label_id in mapping:
                raise ValueError(
                    f"Label ID {label_id} is defined more than once in class_mapping.json."
                )
            mapping[label_id] = raw_class

    if len(mapping) == 0:
        raise ValueError("class_mapping.json must contain at least one class.")
    return mapping


def _parse_mapping_key(raw_label_id: str) -> List[int]:
    try:
        return [int(raw_label_id)]
    except (TypeError, ValueError):
        pass

    if not isinstance(raw_label_id, str) or raw_label_id.count("-") != 1:
        raise ValueError(
            "class_mapping.json keys must be numeric label IDs or inclusive ranges. "
            f"Got {raw_label_id!r}."
        )

    start_raw, end_raw = raw_label_id.split("-")
    try:
        start = int(start_raw)
        end = int(end_raw)
    except ValueError:
        raise ValueError(
            "class_mapping.json range keys must have numeric bounds. "
            f"Got {raw_label_id!r}."
        )
    if start > end:
        raise ValueError(
            f"class_mapping.json range start must be less than or equal to end. Got {raw_label_id!r}."
        )
    return list(range(start, end + 1))


def read_pcd_label_indices(pcd_path: str, class_mapping: Dict[int, str]) -> Dict[str, List[int]]:
    try:
        header, data_offset = _read_pcd_header(pcd_path)
        if header.get("DATA", [""])[0].lower() not in {"ascii", "binary"}:
            return {}
        fields = header.get("FIELDS")
        sizes = _parse_ints(header.get("SIZE"))
        types = header.get("TYPE")
        counts = _parse_ints(header.get("COUNT"))
        points = _parse_int(header.get("POINTS", [None])[0])
        if not fields or not sizes or not types or points is None:
            return {}
        if counts is None:
            counts = [1] * len(fields)
        if (
            len(fields) != len(sizes)
            or len(fields) != len(types)
            or len(fields) != len(counts)
        ):
            return {}
        if "labels" not in fields:
            return {}

        label_index = fields.index("labels")
        if counts[label_index] != 1:
            logger.warning(
                f"Skipping labels in PCD file {pcd_path}: COUNT for labels must be 1."
            )
            return {}

        data_format = header["DATA"][0].lower()
        if data_format == "ascii":
            label_column = sum(counts[:label_index])
            labels = _read_ascii_labels(pcd_path, data_offset, points, label_column)
        else:
            label_offset = sum(
                size * count for size, count in zip(sizes[:label_index], counts[:label_index])
            )
            point_step = sum(size * count for size, count in zip(sizes, counts))
            labels = _read_binary_labels(
                pcd_path,
                data_offset,
                points,
                point_step,
                label_offset,
                sizes[label_index],
                types[label_index],
            )

        indices_by_class = defaultdict(list)
        for point_index, label_id in enumerate(labels):
            class_name = class_mapping.get(label_id)
            if class_name is not None:
                indices_by_class[class_name].append(point_index)
        return dict(indices_by_class)
    except Exception as e:
        logger.debug(f"Failed to read labels from PCD file {pcd_path}: {repr(e)}")
        return {}


def _read_pcd_header(pcd_path: str) -> tuple:
    header = {}
    with open(pcd_path, "rb") as file:
        while True:
            line = file.readline()
            if line == b"":
                break
            decoded = line.decode("ascii", errors="replace").strip()
            if decoded == "" or decoded.startswith("#"):
                continue
            parts = decoded.split()
            header[parts[0]] = parts[1:]
            if parts[0] == "DATA":
                return header, file.tell()
    return header, 0


def _read_ascii_labels(
    pcd_path: str,
    data_offset: int,
    points: int,
    label_column: int,
) -> List[int]:
    labels = []
    with open(pcd_path, "rb") as file:
        file.seek(data_offset)
        for _ in range(points):
            line = file.readline()
            if line == b"":
                break
            parts = line.decode("ascii", errors="replace").split()
            if len(parts) <= label_column:
                continue
            labels.append(int(float(parts[label_column])))
    return labels


def _read_binary_labels(
    pcd_path: str,
    data_offset: int,
    points: int,
    point_step: int,
    label_offset: int,
    label_size: int,
    label_type: str,
) -> List[int]:
    labels = []
    with open(pcd_path, "rb") as file:
        file.seek(data_offset)
        for _ in range(points):
            point = file.read(point_step)
            if len(point) < point_step:
                break
            labels.append(_unpack_pcd_scalar(point, label_offset, label_size, label_type))
    return labels


def _unpack_pcd_scalar(data: bytes, offset: int, size: int, scalar_type: str) -> int:
    formats = {
        ("U", 1): "B",
        ("U", 2): "H",
        ("U", 4): "I",
        ("U", 8): "Q",
        ("I", 1): "b",
        ("I", 2): "h",
        ("I", 4): "i",
        ("I", 8): "q",
        ("F", 4): "f",
        ("F", 8): "d",
    }
    fmt = formats.get((scalar_type, size))
    if fmt is None:
        raise ValueError(f"Unsupported PCD labels scalar type: TYPE={scalar_type}, SIZE={size}.")
    value = struct.unpack_from("<" + fmt, data, offset)[0]
    return int(value)


def _parse_ints(values: Optional[List[str]]) -> Optional[List[int]]:
    if values is None:
        return None
    return [int(value) for value in values]


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)
