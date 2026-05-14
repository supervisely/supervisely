# coding: utf-8

from __future__ import annotations

from typing import List, Optional

from supervisely.geometry.constants import (
    CLASS_ID,
    CREATED_AT,
    GEOMETRY_SHAPE,
    GEOMETRY_TYPE,
    ID,
    INDICES,
    LABELER_LOGIN,
    UPDATED_AT,
)
from supervisely.geometry.geometry import Geometry


class Mesh(Geometry):
    """Mesh geometry represented by selected mesh vertex/face indices."""

    @staticmethod
    def geometry_name():
        return "mesh"

    def __init__(
        self,
        indices: Optional[List[int]] = None,
        sly_id=None,
        class_id=None,
        labeler_login=None,
        updated_at=None,
        created_at=None,
    ):
        super().__init__(
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
        if indices is None:
            indices = []
        if type(indices) is not list:
            raise TypeError('"indices" param has to be of type list, got {!r}'.format(type(indices)))
        self._indices = indices

    @property
    def indices(self):
        return self._indices.copy()

    def to_json(self):
        res = {
            INDICES: self.indices,
            GEOMETRY_SHAPE: self.geometry_name(),
            GEOMETRY_TYPE: self.geometry_name(),
        }
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, data):
        indices = data.get(INDICES)
        if indices is not None and not isinstance(indices, list):
            raise ValueError(
                f"Mesh '{INDICES}' field must be a list, got {type(indices).__name__!r}."
            )
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return cls(
            indices or [],
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    def to_bytes(self) -> bytes:
        """Encode mesh indices as little-endian uint32 bytes."""
        from supervisely.mesh_annotation.mesh_indices import encode_mesh_indices

        return encode_mesh_indices(self._indices)

    @classmethod
    def from_bytes(cls, data: bytes) -> Mesh:
        """Create a Mesh from little-endian uint32 bytes."""
        from supervisely.mesh_annotation.mesh_indices import decode_mesh_indices

        return cls(indices=decode_mesh_indices(data))

    @classmethod
    def from_file(cls, file_path: str) -> Mesh:
        with open(file_path, "rb") as f:
            return cls.from_bytes(f.read())