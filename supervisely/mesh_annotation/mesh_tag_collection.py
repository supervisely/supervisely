# coding: utf-8
from __future__ import annotations

from typing import Iterator

from supervisely.mesh_annotation.mesh_tag import MeshTag
from supervisely.pointcloud_annotation.pointcloud_tag_collection import PointcloudTagCollection


class MeshTagCollection(PointcloudTagCollection):
    """Collection of :class:`MeshTag` instances."""

    item_type = MeshTag

    def __iter__(self) -> Iterator[MeshTag]:
        """Iterate over the :class:`MeshTag` instances in the collection."""
        return next(self)
