# coding: utf-8

from supervisely_lib.api.module_api import ApiField
from supervisely_lib.api.entity_annotation.tag_api import TagApi


class VideoTagApi(TagApi):
    _entity_id_field = ApiField.VIDEO_ID
    _method_bulk_add = 'videos.tags.bulk.add'
