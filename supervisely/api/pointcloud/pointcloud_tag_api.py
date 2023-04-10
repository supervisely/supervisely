# coding: utf-8

from supervisely.api.module_api import ApiField
from supervisely.api.entity_annotation.tag_api import TagApi


class PointcloudTagApi(TagApi):
    """
    :class:`PointcloudTag<supervisely.pointcloud_annotation.pointcloud_tag.PointcloudTag>` for point clouds. :class:`PointcloudTagApi<PointcloudTagApi>` object is immutable.
    """

    _entity_id_field = ApiField.ENTITY_ID
    _method_bulk_add = 'point-clouds.tags.bulk.add'

