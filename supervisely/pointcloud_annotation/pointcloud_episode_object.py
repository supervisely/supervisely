# coding: utf-8

# docs
from supervisely.pointcloud_annotation.pointcloud_episode_tag_collection import (
    PointcloudEpisodeTagCollection,
)
from supervisely.pointcloud_annotation.pointcloud_object import PointcloudObject


class PointcloudEpisodeObject(PointcloudObject):
    """
    Object tracked across point cloud episode frames (obj_class + tags). Immutable.

    :param obj_class: Object class.
    :type obj_class: :class:`~supervisely.annotation.obj_class.ObjClass`
    :param tags: Pointcloud episode tags collection.
    :type tags: :class:`~supervisely.pointcloud_annotation.pointcloud_episode_tag_collection.PointcloudEpisodeTagCollection`
    :param key: UUID key associated with the object.
    :type key: uuid.UUID
    :param class_id: ID of ObjClass to which PointcloudEpisodeObject belongs.
    :type class_id: int
    :param labeler_login: Login of the user who created :class:`~supervisely.pointcloud_annotation.pointcloud_episode_object.PointcloudEpisodeObject`.
    :type labeler_login: str
    :param updated_at: Date and Time when PointcloudEpisodeObject was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str
    :param created_at: Date and Time when PointcloudEpisodeObject was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str

    :Usage Example:

        .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid_3d import Cuboid3d

            obj_class_car = sly.ObjClass('car', Cuboid3d)
            pointcloud_episode_obj_car = sly.PointcloudEpisodeObject(obj_class_car)
            pointcloud_episode_obj_car_json = pointcloud_episode_obj_car.to_json()
            print(pointcloud_episode_obj_car_json)
            # Output: {
            #     "key": "6b819f1840f84d669b32cdec225385f0",
            #     "classTitle": "car",
            #     "tags": []
            # }
    """

    tag_collection_type = PointcloudEpisodeTagCollection
