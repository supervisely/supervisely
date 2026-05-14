# coding: utf-8
from __future__ import annotations

from typing import Dict, Iterator, List, Optional

from supervisely.mesh_annotation.mesh_object import MeshObject
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.key_id_map import KeyIdMap


class MeshObjectCollection(PointcloudObjectCollection):
    """Collection of :class:`MeshObject` instances."""

    item_type = MeshObject

    def __iter__(self) -> Iterator[MeshObject]:
        return next(self)

    @classmethod
    def from_json(
        cls, data: List[Dict], project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> "MeshObjectCollection":
        return super().from_json(data, project_meta, key_id_map=key_id_map)
