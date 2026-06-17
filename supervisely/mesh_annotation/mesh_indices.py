# coding: utf-8
from __future__ import annotations

import base64
from copy import deepcopy
from typing import Any, Iterable, List

import numpy as np


MESH_INDEX_FIELDS = (
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
)


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
    if len(data) % 4 != 0:
        raise ValueError("Encoded mesh indices byte length must be divisible by 4.")
    return np.frombuffer(data, dtype="<u4").tolist()


def encode_mesh_indices(indices: Iterable[int]) -> bytes:
    """Encode mesh indices as little-endian uint32 bytes.

    :param indices: Sequence of non-negative integer mesh indices.
    :type indices: Iterable[int]
    :returns: Little-endian uint32 byte representation of the indices.
    :rtype: bytes
    """
    return encode_mesh_indices_np(indices)


def decode_mesh_indices(data: bytes) -> List[int]:
    """Decode little-endian uint32 mesh index bytes into a list of integers.

    :param data: Little-endian uint32 encoded mesh indices.
    :type data: bytes
    :returns: Decoded mesh indices.
    :rtype: List[int]
    """
    return decode_mesh_indices_np(data)


def encode_mesh_indices_base64(indices: Iterable[int]) -> str:
    """Encode mesh indices as a base64 string of little-endian uint32 bytes.

    :param indices: Sequence of non-negative integer mesh indices.
    :type indices: Iterable[int]
    :returns: Base64-encoded mesh indices.
    :rtype: str
    """
    return base64.b64encode(encode_mesh_indices(indices)).decode("ascii")


def decode_mesh_indices_base64(data: str) -> List[int]:
    """Decode a base64 string of little-endian uint32 bytes into a list of integers.

    :param data: Base64-encoded mesh indices.
    :type data: str
    :returns: Decoded mesh indices.
    :rtype: List[int]
    """
    return decode_mesh_indices(base64.b64decode(data.encode("ascii")))


def encode_mesh_indices_in_json(data: Any) -> Any:
    """Return a JSON copy with mesh index arrays encoded to base64 strings.

    Prefer raw ``encode_mesh_indices`` bytes for API/storage paths. This helper is only for JSON-only
    interchange surfaces.
    """
    return _convert_mesh_indices(deepcopy(data), encode=True)


def decode_mesh_indices_in_json(data: Any) -> Any:
    """Return a JSON copy with base64 mesh index strings decoded to integer arrays."""
    return _convert_mesh_indices(deepcopy(data), encode=False)


def _convert_mesh_indices(data: Any, encode: bool) -> Any:
    """Recursively encode/decode mesh index fields in a nested JSON structure.

    When ``encode`` is True, integer-sequence values under known mesh index fields are
    converted to base64 strings; when False, base64 strings are decoded back to integer lists.
    """
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
    """Return True if ``value`` is a list/tuple/ndarray of integers (not a str/bytes)."""
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.integer)
    if not isinstance(value, (list, tuple)):
        return False
    return all(isinstance(item, (int, np.integer)) for item in value)
