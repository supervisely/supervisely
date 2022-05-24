# coding: utf-8

from supervisely.api.module_api import ApiField
from supervisely.api.entity_annotation.tag_api import TagApi


class VolumeTagApi(TagApi):
    _entity_id_field = ApiField.ENTITY_ID
    _method_bulk_add = "volumes.tags.bulk.add"

    def remove_from_volume(self, tag_id):
        self._api.post("volumes.tags.remove", {ApiField.ID: tag_id})

    def update_value(self, tag_id, tag_value):
        self._api.post(
            "volumes.tags.update-value",
            {ApiField.ID: tag_id, ApiField.VALUE: tag_value},
        )
