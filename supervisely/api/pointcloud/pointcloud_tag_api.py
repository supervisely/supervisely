# coding: utf-8

from supervisely.api.module_api import ApiField
from supervisely.api.entity_annotation.tag_api import TagApi
from typing import List, Optional, Union


class PointcloudTagApi(TagApi):
    _entity_id_field = ApiField.ENTITY_ID
    _method_bulk_add = "point-clouds.tags.bulk.add"

    def add_tag(
        self,
        tag_id: int,
        object_id: int,
        value: Optional[Union[str, int]] = None,
        frame_range: Optional[List[int]] = None,
    ) -> int:
        """Add a tag directly to an annotation object.
        It is possible to add a value as an option for point clouds and point cloud episodes.
        It is also possible to add frames as an option, but only for point cloud episodes.

        :param tag_id: TagMeta ID in project tag_metas
        :type tag_id: int
        :param object_id: Object ID in project annotation objects
        :type object_id: int
        :param value: possible_values from TagMeta, defaults to None
        :type value: Optional[Union[str, int]], optional
        :param frame_range: array of 2 frame numbers in point cloud episodes, defaults to None
        :type frame_range: Optional[List[int]], optional
        :return: ID of the tag assigned to the object
        :rtype: int
        """
        if value is not None and frame_range is not None:
            response = self._api.post(
                "annotation-objects.tags.add",
                {
                    ApiField.TAG_ID: tag_id,
                    ApiField.OBJECT_ID: object_id,
                    ApiField.VALUE: value,
                    ApiField.FRAME_RANGE: frame_range,
                },
            )
        elif value is not None:
            response = self._api.post(
                "annotation-objects.tags.add",
                {
                    ApiField.TAG_ID: tag_id,
                    ApiField.OBJECT_ID: object_id,
                    ApiField.VALUE: value,
                },
            )
        elif frame_range is not None:
            response = self._api.post(
                "annotation-objects.tags.add",
                {
                    ApiField.TAG_ID: tag_id,
                    ApiField.OBJECT_ID: object_id,
                    ApiField.FRAME_RANGE: frame_range,
                },
            )
        else:
            response = self._api.post(
                "annotation-objects.tags.add",
                {
                    ApiField.TAG_ID: tag_id,
                    ApiField.OBJECT_ID: object_id,
                },
            )
        id = response.json()[ApiField.ID]
        return id
