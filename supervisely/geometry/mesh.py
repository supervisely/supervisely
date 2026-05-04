# coding: utf-8

from __future__ import annotations

from typing import List, Optional

from supervisely.geometry.constants import CLASS_ID, CREATED_AT, ID, INDICES, LABELER_LOGIN, UPDATED_AT
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
        res = {INDICES: self.indices}
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, data):
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return cls(
            data.get(INDICES, []),
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
