# coding: utf-8

from supervisely_lib.api.module_api import ApiField, ModuleApiBase
from typing import List, Tuple, Optional, Dict


class AdvancedApi(ModuleApiBase):
    def add_tag_to_object(self, tag_meta_id: int, figure_id: int, value: Optional[str or int]=None) -> Dict:
        data = {ApiField.TAG_ID: tag_meta_id, ApiField.FIGURE_ID: figure_id}
        if value is not None:
            data[ApiField.VALUE] = value
        resp = self._api.post('object-tags.add-to-object', data)
        return resp.json()

    def remove_tag_from_object(self, tag_meta_id: int, figure_id: int, tag_id: int) -> Dict:
        data = {ApiField.TAG_ID: tag_meta_id, ApiField.FIGURE_ID: figure_id, ApiField.ID: tag_id}
        resp = self._api.post('object-tags.remove-from-figure', data)
        return resp.json()

    def get_object_tags(self, figure_id: int) -> Dict:
        data = {ApiField.ID: figure_id}
        resp = self._api.post('figures.tags.list', data)
        return resp.json()

    def remove_tag_from_image(self, tag_meta_id: int, image_id: int, tag_id: int) -> Dict:
        data = {ApiField.TAG_ID: tag_meta_id, ApiField.IMAGE_ID: image_id, ApiField.ID: tag_id}
        resp = self._api.post('image-tags.remove-from-image', data)
        return resp.json()
