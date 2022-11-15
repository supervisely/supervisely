# coding: utf-8
from __future__ import annotations
from typing import Optional, Dict
from uuid import UUID

from supervisely.geometry.geometry import Geometry
from supervisely._utils import take_with_default
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.pointcloud_annotation.pointcloud_object import PointcloudObject
from supervisely.video_annotation.video_figure import VideoFigure
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)


class PointcloudFigure(VideoFigure):
    """
    Class for creating and using PointcloudFigure objects for pointclouds
    """

    def __init__(
        self,
        parent_object: PointcloudObject,
        geometry: Geometry,
        frame_index: Optional[int] = None,
        key: Optional[UUID] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        """
        :param parent_object: PointcloudObject class object
        :param geometry: Geometry class object (cuboid_3d only)
        :param frame_index: int
        :param key: uuid class object
        """
        super().__init__(
            parent_object,
            geometry,
            frame_index,
            key=key,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
        # @TODO: validate geometry - allowed: only cuboid_3d + point_cloud

    @property
    def parent_object(self) -> PointcloudObject:
        return super().parent_object

    @property
    def video_object(self) -> None:
        raise NotImplementedError("If you faced this error, please write to technical support.")

    def validate_bounds(self, img_size, _auto_correct=False):
        raise NotImplementedError()

    @classmethod
    def from_json(
        cls,
        data: Dict,
        objects: PointcloudObjectCollection,
        frame_index: Optional[int] = None,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> PointcloudFigure:
        return super().from_json(data, objects, frame_index, key_id_map)

    def clone(
        self,
        parent_object: Optional[PointcloudObject] = None,
        geometry: Optional[Geometry] = None,
        frame_index: Optional[int] = None,
        key: Optional[UUID] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> PointcloudFigure:
        return super().clone(
            video_object=take_with_default(parent_object, self.parent_object),
            geometry=take_with_default(geometry, self.geometry),
            frame_index=take_with_default(frame_index, self.frame_index),
            key=take_with_default(key, self._key),
            class_id=take_with_default(class_id, self.class_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at),
        )
