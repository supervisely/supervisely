# coding: utf-8
from __future__ import annotations

import base64
from copy import deepcopy
from typing import Any, Iterable, List

import numpy as np


MESH_INDEX_FIELDS = {
    "indices",
    "vertexIndices",
    "verticesIndices",
    "edgeIndices",
    "edgesIndices",
    "faceIndices",
    "facesIndices",
    "triangleIndices",
    "trianglesIndices",
    "vertex_indices",
    "vertices_indices",
    "edge_indices",
    "edges_indices",
    "face_indices",
    "faces_indices",
    "triangle_indices",
    "triangles_indices",
}


def encode_mesh_indices_np(indices: Iterable[int]) -> bytes:
    """Encode mesh indices as little-endian uint32 bytes."""
    arr = indices if isinstance(indices, np.ndarray) else np.asarray(list(indices))
    if arr.size == 0:
        return b""
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("Mesh indices must be integers.")
    if np.any(arr < 0):
        raise ValueError("Mesh indices must be non-negative.")
    if np.any(arr > np.iinfo(np.uint32).max):
        raise ValueError("Mesh indices must fit into uint32.")
    return arr.astype("<u4", copy=False).tobytes()


def decode_mesh_indices_np(data: bytes) -> List[int]:
    """Decode little-endian uint32 mesh index bytes."""
    aligned_len = len(data) - (len(data) % 4)
    return np.frombuffer(data[:aligned_len], dtype="<u4").tolist()


def encode_mesh_indices(indices: Iterable[int]) -> bytes:
    return encode_mesh_indices_np(indices)


def decode_mesh_indices(data: bytes) -> List[int]:
    return decode_mesh_indices_np(data)


def encode_mesh_indices_base64(indices: Iterable[int]) -> str:
    return base64.b64encode(encode_mesh_indices(indices)).decode("ascii")


def decode_mesh_indices_base64(data: str) -> List[int]:
    return decode_mesh_indices(base64.b64decode(data.encode("ascii")))


def encode_mesh_indices_in_json(data: Any) -> Any:
    """Return a JSON copy with mesh index arrays encoded to base64 strings."""
    return _convert_mesh_indices(deepcopy(data), encode=True)


def decode_mesh_indices_in_json(data: Any) -> Any:
    """Return a JSON copy with base64 mesh index strings decoded to integer arrays."""
    return _convert_mesh_indices(deepcopy(data), encode=False)


def _convert_mesh_indices(data: Any, encode: bool) -> Any:
    if isinstance(data, dict):
        for key, value in list(data.items()):
            if key in MESH_INDEX_FIELDS:
                if encode and _is_int_sequence(value):
                    data[key] = encode_mesh_indices_base64(value)
                    continue
                if not encode and isinstance(value, str):
                    data[key] = decode_mesh_indices_base64(value)
                    continue
            data[key] = _convert_mesh_indices(value, encode)
        return data

    if isinstance(data, list):
        return [_convert_mesh_indices(item, encode) for item in data]

    return data


def _is_int_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.integer)
    if not isinstance(value, (list, tuple)):
        return False
    return all(isinstance(item, (int, np.integer)) for item in value)
