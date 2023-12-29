# coding: utf-8
from __future__ import annotations
from typing import List, Dict, Optional, Iterator
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)
from supervisely.pointcloud_annotation.pointcloud_episode_object import PointcloudEpisodeObject
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.key_id_map import KeyIdMap


class PointcloudEpisodeObjectCollection(PointcloudObjectCollection):
    """
    Collection with :class:`PointcloudEpisodeObject<supervisely.pointcloud_annotation.pointcloud_episode_object.PointcloudEpisodeObject>` instances.
    :class:`PointcloudEpisodeObjectCollection<PointcloudEpisodeObjectCollection>` object is immutable.
    """

    item_type = PointcloudEpisodeObject

    def __iter__(self) -> Iterator[PointcloudEpisodeObject]:
        return next(self)

    @classmethod
    def from_json(
        cls, data: List[Dict], project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> PointcloudEpisodeObjectCollection:
        """
        Convert a list of json dicts to PointcloudEpisodeObjectCollection. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: List with dicts in json format.
        :type data: List[dict]
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`.
        :type project_meta: ProjectMeta
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: PointcloudEpisodeObjectCollection object
        :rtype: :class:`PointcloudEpisodeObjectCollection`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.cuboid_3d import Cuboid3d
            from supervisely.pointcloud_annotation.pointcloud_episode_object_collection import PointcloudEpisodeObjectCollection

            obj_collection_json = [
                {
                    "classTitle": "car",
                    "tags": []
                },
                {
                    "classTitle": "bus",
                    "tags": []
                }
            ]

            class_car = sly.ObjClass('car', Cuboid3d)
            class_bus = sly.ObjClass('bus', Cuboid3d)
            classes = sly.ObjClassCollection([class_car, class_bus])
            meta = sly.ProjectMeta(obj_classes=classes)

            pointcloud_obj_collection = sly.PointcloudEpisodeObjectCollection.from_json(obj_collection_json, meta)
        """

        return super().from_json(data, project_meta, key_id_map=key_id_map)
