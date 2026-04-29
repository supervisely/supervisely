# coding: utf-8
from __future__ import annotations

from supervisely.mesh_annotation.mesh_tag_collection import MeshTagCollection
from supervisely.pointcloud_annotation.pointcloud_object import PointcloudObject


class MeshObject(PointcloudObject):
    """Object in a mesh annotation."""

    tag_collection_type = MeshTagCollection
