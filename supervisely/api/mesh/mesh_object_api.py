# coding: utf-8
from __future__ import annotations

from typing import Iterable, List

from supervisely.api.entity_annotation.object_api import ObjectApi
from supervisely.api.mesh.mesh_tag_api import MeshObjectTagApi
from supervisely.video_annotation.key_id_map import KeyIdMap


class MeshObjectApi(ObjectApi):
    """Internal API for mesh annotation object rows."""

    def __init__(self, api):
        super().__init__(api)
        self.tag = MeshObjectTagApi(api)

    def append_bulk(
        self,
        mesh_id: int,
        objects: Iterable,
        key_id_map: KeyIdMap = None,
    ) -> List[int]:
        info = self._api.mesh.get_info_by_id(mesh_id)
        return self._append_bulk(
            self._api.mesh.tag,
            mesh_id,
            info.project_id,
            info.dataset_id,
            objects,
            key_id_map,
        )
