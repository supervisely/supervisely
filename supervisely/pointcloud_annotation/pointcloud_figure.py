# coding: utf-8
from __future__ import annotations

from typing import Dict, Optional, Union
from uuid import UUID

from supervisely._utils import take_with_default
from supervisely.geometry.geometry import Geometry
from supervisely.pointcloud_annotation.pointcloud_episode_object import (
    PointcloudEpisodeObject,
)
from supervisely.pointcloud_annotation.pointcloud_episode_object_collection import (
    PointcloudEpisodeObjectCollection,
)
from supervisely.pointcloud_annotation.pointcloud_object import PointcloudObject
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_figure import VideoFigure


class PointcloudFigure(VideoFigure):
    """
    PointcloudFigure object for
    :class:`PointcloudAnnotation<supervisely.pointcloud_annotation.pointcloud_annotation.PointcloudAnnotation>` or :class:`PointcloudEpisodeAnnotation<supervisely.pointcloud_annotation.pointcloud_episode_annotation.PointcloudEpisodeAnnotation>`.
    :class:`PointcloudFigure<PointcloudFigure>` objects is immutable.

    :param parent_object: PointcloudObject or PointcloudObject object.
    :type parent_object: Union[PointcloudObject, PointcloudEpisodeObject]
    :param geometry: Label :class:`geometry<supervisely.geometry.geometry.Geometry>`.
    :type geometry: Geometry
    :param frame_index: Index of Frame to which PointcloudFigure belongs.
    :type frame_index: int
    :param key: KeyIdMap object.
    :type key: KeyIdMap, optional
    :param class_id: ID of :class:`PointcloudObject<PointcloudObject>` (or :class:`PointcloudEpisodeObject<PointcloudEpisodeObject>`) to which PointcloudFigure belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created PointcloudFigure.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when PointcloudFigure was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when PointcloudFigure was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d

        obj_class_car = sly.ObjClass('car', Cuboid3d)
        pointcloud_obj_car = sly.PointcloudObject(obj_class_car)

        position, rotation, dimension = Vector3d(-3.4, 28.9, -0.7), Vector3d(0., 0, -0.03), Vector3d(1.8, 3.9, 1.6)
        cuboid = Cuboid3d(position, rotation, dimension)
        frame_index = 10
        figure = sly.PointcloudFigure(pointcloud_obj_car, cuboid, frame_index=frame_index)

        print(figure.to_json())
        # Output: {
        #     "geometry": {
        #         "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
        #         "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
        #         "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
        #     },
        #     "geometryType": "cuboid_3d",
        #     "key": "4beae1be12624b70ad533c8be7477605",
        #     "objectKey": "c1e1965efc0d4ae9b0b39367b04d637a"
        # }
    """

    def __init__(
        self,
        parent_object: Union[PointcloudObject, PointcloudEpisodeObject],
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
    def parent_object(self) -> Union[PointcloudObject, PointcloudEpisodeObject]:
        """
        PointcloudObject of current PointcloudFigure.

        :return: PointcloudObject ot PointcloudEpisodeObject object
        :rtype: :class:`PointcloudObject<PointcloudObject>` or :class:`PointcloudEpisodeObject<PointcloudEpisodeObject>`
        :Usage example:
         .. code-block:: python

            pointcloud_obj_car = pointcloud_figure_car.parent_object

            print(pointcloud_obj_car.to_json())
            # Output: {
            #     "key": "d573c6f081544e3da20022d932b259c1",
            #     "classTitle": "car",
            #     "tags": []
            # }
        """

        return super().parent_object

    @property
    def video_object(self) -> None:
        """Not supported for pointcloud."""

        raise NotImplementedError("If you faced this error, please write to technical support.")

    def validate_bounds(self, img_size, _auto_correct=False):
        """Not supported for pointcloud."""

        raise NotImplementedError()

    @classmethod
    def from_json(
        cls,
        data: Dict,
        objects: Union[PointcloudObjectCollection, PointcloudEpisodeObjectCollection],
        frame_index: Optional[int] = None,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> PointcloudFigure:
        """
        Convert a json dict to PointcloudFigure. Read more about `Supervisely format <https://docs.supervisely.com/data-organization/00_ann_format_navi>`_.

        :param data: Dict in json format.
        :type data: dict
        :param objects: PointcloudObjectCollection or PointcloudEpisodeObjectCollection object.
        :type objects: PointcloudObjectCollection or PointcloudEpisodeObjectCollection
        :param frame_index: Index of Frame to which PointcloudFigure belongs.
        :type frame_index: int
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :raises: :class:`RuntimeError`, if point cloudobject ID and pointcloud object key are None, if pointcloud object key and key_id_map are None, if pointcloud object with given id not found in key_id_map
        :return: PointcloudFigure object
        :rtype: :class:`PointcloudFigure`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d

            obj_class_car = sly.ObjClass('car', Cuboid3d)
            pointcloud_obj_car = sly.PointcloudObject(obj_class_car)

            position, rotation, dimension = Vector3d(-3.4, 28.9, -0.7), Vector3d(0., 0, -0.03), Vector3d(1.8, 3.9, 1.6)
            cuboid = Cuboid3d(position, rotation, dimension)
            frame_index = 10
            figure = sly.PointcloudFigure(pointcloud_obj_car, cuboid, frame_index=frame_index)
            pointcloud_figure_json = figure.to_json(save_meta=True)

            new_pointcloud_figure = sly.PointcloudFigure.from_json(
                pointcloud_figure_json,
                sly.PointcloudObjectCollection([pointcloud_obj_car]),
                frame_index
            )
        """

        return super().from_json(data, objects, frame_index, key_id_map)

    def clone(
        self,
        parent_object: Optional[Union[PointcloudObject, PointcloudEpisodeObject]] = None,
        geometry: Optional[Geometry] = None,
        frame_index: Optional[int] = None,
        key: Optional[UUID] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> PointcloudFigure:
        """
        Makes a copy of PointcloudFigure with new fields, if fields are given, otherwise it will use fields of the original PointcloudFigure.

        :param parent_object: :class:`PointcloudObject<PointcloudObject>` (or :class:`PointcloudEpisodeObject<PointcloudEpisodeObject>`) object.
        :type parent_object: PointcloudObject or PointcloudEpisodeObject, optional
        :param geometry: Label :class:`geometry<supervisely.geometry.geometry.Geometry>`.
        :type geometry: Geometry, optional
        :param frame_index: Index of Frame to which PointcloudFigure belongs.
        :type frame_index: int, optional
        :param key: KeyIdMap object.
        :type key: KeyIdMap, optional
        :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which PointcloudFigure belongs.
        :type class_id: int, optional
        :param labeler_login: Login of the user who created PointcloudFigure.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when PointcloudFigure was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: str, optional
        :param created_at: Date and Time when PointcloudFigure was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: str, optional
        :return: PointcloudFigure object
        :rtype: :class:`PointcloudFigure`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d

            obj_class_car = sly.ObjClass('car', Cuboid3d)
            pointcloud_obj_car = sly.PointcloudObject(obj_class_car)

            position, rotation, dimension = Vector3d(-3.4, 28.9, -0.7), Vector3d(0., 0, -0.03), Vector3d(1.8, 3.9, 1.6)
            cuboid = Cuboid3d(position, rotation, dimension)
            frame_index = 10
            figure = sly.PointcloudFigure(pointcloud_obj_car, cuboid, frame_index=frame_index)

            # Remember that PointcloudFigure object is immutable, and we need to assign new instance of PointcloudFigure to a new variable
            pointcloud_figure_clone = figure.clone(parent_object=pointcloud_obj_car, frame_index=11)

            print(pointcloud_figure_clone.to_json())
            # Output: {
            #     "geometry": {
            #         "dimensions": { "x": 1.8, "y": 3.9, "z": 1.6 },
            #         "position": { "x": -3.4, "y": 28.9, "z": -0.7 },
            #         "rotation": { "x": 0.0, "y": 0, "z": -0.03 }
            #     },
            #     "geometryType": "cuboid_3d",
            #     "key": "4beae1be12624b70ad533c8be7477605",
            #     "objectKey": "c1e1965efc0d4ae9b0b39367b04d637a"
            # }
        """

        return self.__class__(
            parent_object=take_with_default(parent_object, self.parent_object),
            geometry=take_with_default(geometry, self.geometry),
            frame_index=take_with_default(frame_index, self.frame_index),
            key=take_with_default(key, self._key),
            class_id=take_with_default(class_id, self.class_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at),
        )
