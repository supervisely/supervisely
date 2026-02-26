from __future__ import annotations

from typing import Dict, List, Optional

from supervisely._utils import take_with_default
from supervisely.pointcloud_annotation.pointcloud_episode_object_collection import (
    PointcloudEpisodeObjectCollection,
)
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.video_annotation.frame import Frame
from supervisely.video_annotation.key_id_map import KeyIdMap


class PointcloudEpisodeFrame(Frame):
    """Single frame in point cloud episode; holds PointcloudFigures at given index. Immutable."""

    figure_type = PointcloudFigure

    def __init__(self, index: int, figures: Optional[List[PointcloudFigure]] = None):
        """
        Same parameters as :class:`~supervisely.video_annotation.frame.Frame`.

        :param index: Frame index.
        :type index: int
        :param figures: List of PointcloudFigures.
        :type figures: List[:class:`~supervisely.pointcloud_annotation.pointcloud_figure.PointcloudFigure`], optional

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d

                obj_class_car = sly.ObjClass('car', Cuboid3d)
                pointcloud_obj_car = sly.PointcloudEpisodeObject(obj_class_car)
                cuboid = Cuboid3d(
                    Vector3d(-3.4, 28.9, -0.7), Vector3d(0., 0, -0.03), Vector3d(1.8, 3.9, 1.6)
                )
                figure = sly.PointcloudFigure(pointcloud_obj_car, cuboid, frame_index=7)
                frame = sly.PointcloudEpisodeFrame(7, figures=[figure])
                print(frame.to_json())
        """
        super().__init__(index, figures)

    @classmethod
    def from_json(
        cls,
        data: Dict,
        objects: PointcloudEpisodeObjectCollection,
        frames_count: Optional[int] = None,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> PointcloudEpisodeFrame:
        """
        Convert a json dict to PointcloudEpisodeFrame. Read more about `Supervisely format <https://docs.supervisely.com/data-organization/00_ann_format_navi>`_.

        :param data: Dict in json format.
        :type data: dict
        :param objects: Pointcloud episode objects collection.
        :type objects: :class:`~supervisely.pointcloud_annotation.pointcloud_episode_object_collection.PointcloudEpisodeObjectCollection`
        :param frames_count: Number of frames in pointcloud episode.
        :type frames_count: int
        :param key_id_map: Key ID map.
        :type key_id_map: :class:`~supervisely.video_annotation.key_id_map.KeyIdMap`
        :raises ValueError: if frame index < 0 and if frame index > number of frames in pointcloud episode
        :returns: Pointcloud episode frame object.
        :rtype: :class:`~supervisely.pointcloud_annotation.pointcloud_episode_frame.PointcloudEpisodeFrame`

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
                from supervisely.pointcloud_annotation.pointcloud_episode_object_collection import PointcloudEpisodeObjectCollection

                obj_class_car = sly.ObjClass('car', Cuboid3d)
                pointcloud_obj_car = sly.PointcloudEpisodeObject(obj_class_car)
                objects = PointcloudEpisodeObjectCollection([pointcloud_obj_car])

                frame_index = 7
                position, rotation, dimension = Vector3d(-3.4, 28.9, -0.7), Vector3d(0., 0, -0.03), Vector3d(1.8, 3.9, 1.6)
                cuboid = Cuboid3d(position, rotation, dimension)

                figure = sly.PointcloudFigure(pointcloud_obj_car, cuboid, frame_index=frame_index)

                frame = sly.PointcloudEpisodeFrame(frame_index, figures=[figure])
                frame_json = frame.to_json()

                frame_from_json = sly.PointcloudEpisodeFrame.from_json(frame_json, objects)
        """

        return super().from_json(data, objects, frames_count, key_id_map)

    def clone(
        self, index: Optional[int] = None, figures: Optional[List[PointcloudFigure]] = None
    ) -> PointcloudEpisodeFrame:
        return super().clone(index, figures)
