# coding: utf-8
from __future__ import annotations

from typing import Optional
from uuid import UUID

from supervisely.geometry.geometry import Geometry
from supervisely.mesh_annotation.mesh_object import MeshObject
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure


class MeshFigure(PointcloudFigure):
    """Figure in a mesh annotation."""

    def __init__(
        self,
        parent_object: MeshObject,
        geometry: Geometry,
        frame_index: Optional[int] = None,
        key: Optional[UUID] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            parent_object=parent_object,
            geometry=geometry,
            frame_index=frame_index,
            key=key,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
            **kwargs,
        )
