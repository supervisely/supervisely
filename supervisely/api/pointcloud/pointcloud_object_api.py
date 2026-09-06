# coding: utf-8

"""Work with point cloud objects via the Supervisely API."""

from typing import List

from supervisely.api.entity_annotation.object_api import ObjectApi
from supervisely.api.pointcloud.pointcloud_tag_api import PointcloudObjectTagApi
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)
from supervisely.video_annotation.key_id_map import KeyIdMap


class PointcloudObjectApi(ObjectApi):
    """
    API for working with point cloud objects.
    """

    def __init__(self, api):
        """
        :param api: :class:`~supervisely.api.api.Api` object to use for API connection.
        :type api: :class:`~supervisely.api.api.Api`
        """
        super().__init__(api)
        self.tag = PointcloudObjectTagApi(api)

    def append_bulk(
        self,
        pointcloud_id: int,
        objects: PointcloudObjectCollection,
        key_id_map: KeyIdMap = None,
    ) -> List[int]:
        """
        Add point cloud objects to annotation objects.

        :param pointcloud_id: Point cloud ID in Supervisely.
        :type pointcloud_id: int
        :param objects: PointcloudObjectCollection objects collection.
        :type objects: :class:`~supervisely.pointcloud_annotation.pointcloud_object_collection.PointcloudObjectCollection`
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`~supervisely.video_annotation.key_id_map.KeyIdMap`, optional
        :returns: List of object IDs
        :rtype: :class:`List[int]`

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly
                from supervisely.video_annotation.key_id_map import KeyIdMap

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                project_id = 19442
                pointcloud_id = 19618685

                meta_json = api.project.get_meta(project_id)
                project_meta = sly.ProjectMeta.from_json(meta_json)

                key_id_map = KeyIdMap()
                ann_info = api.pointcloud.annotation.download(pointcloud_id)
                ann = sly.PointcloudAnnotation.from_json(ann_info, project_meta, key_id_map)

                res = api.pointcloud.object.append_bulk(pointcloud_id, ann.objects, key_id_map)
                print(res)

                # Output: [5565915, 5565916, 5565917, 5565918, 5565919]
        """

        info = self._api.pointcloud.get_info_by_id(pointcloud_id)
        return self._append_bulk(
            self._api.pointcloud.tag,
            pointcloud_id,
            info.project_id,
            info.dataset_id,
            objects,
            key_id_map,
            is_pointcloud=True,
        )

    def append_to_dataset(
        self,
        dataset_id: int,
        objects: PointcloudObjectCollection,
        key_id_map: KeyIdMap = None,
    ) -> List[int]:
        """
        Add pointcloud objects to Dataset annotation objects.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param objects: PointcloudObjectCollection objects collection.
        :type objects: :class:`~supervisely.pointcloud_annotation.pointcloud_object_collection.PointcloudObjectCollection`
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`~supervisely.video_annotation.key_id_map.KeyIdMap`, optional
        :returns: List of objects IDs
        :rtype: :class:`List[int]`

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly
                from supervisely.pointcloud_annotation.pointcloud_annotation import PointcloudObjectCollection
                from supervisely.video_annotation.key_id_map import KeyIdMap

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                project_id = 19442
                project = api.project.get_info_by_id(project_id)
                dataset = api.dataset.create(project.id, "demo_dataset", change_name_if_conflict=True)

                class_car = sly.ObjClass('car', sly.Cuboid)
                class_pedestrian = sly.ObjClass('pedestrian', sly.Cuboid)
                classes = sly.ObjClassCollection([class_car, class_pedestrian])
                project_meta = sly.ProjectMeta(classes)
                updated_meta = api.project.update_meta(project.id, project_meta.to_json())

                key_id_map = KeyIdMap()

                pedestrian_object = sly.PointcloudObject(class_pedestrian)
                car_object = sly.PointcloudObject(class_car)
                objects_collection = PointcloudObjectCollection([pedestrian_object, car_object])

                uploaded_objects_ids = api.pointcloud_episode.object.append_to_dataset(
                    dataset.id,
                    objects_collection,
                    key_id_map,
                )
                print(uploaded_objects_ids)

                # Output: [5565920, 5565921, 5565922]
        """

        project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
        return self._append_bulk(
            self._api.pointcloud.tag,
            dataset_id,
            project_id,
            dataset_id,
            objects,
            key_id_map,
            is_pointcloud=True,
        )
