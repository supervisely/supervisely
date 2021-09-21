# coding: utf-8

from supervisely_lib.api.module_api import ApiField, ModuleApiBase


class AdvancedApi(ModuleApiBase):
    def add_tag_to_object(self, tag_meta_id, figure_id, value=None):
        data = {ApiField.TAG_ID: tag_meta_id, ApiField.FIGURE_ID: figure_id}
        if value is not None:
            data[ApiField.VALUE] = value
        resp = self._api.post('object-tags.add-to-object', data)
        return resp.json()

    def remove_tag_from_object(self, tag_meta_id, figure_id, tag_id):
        data = {ApiField.TAG_ID: tag_meta_id, ApiField.FIGURE_ID: figure_id, ApiField.ID: tag_id}
        resp = self._api.post('object-tags.remove-from-figure', data)
        return resp.json()

    def get_object_tags(self, figure_id):
        data = {ApiField.ID: figure_id}
        resp = self._api.post('figures.tags.list', data)
        return resp.json()

    def remove_tag_from_image(self, tag_meta_id, image_id, tag_id):
        data = {ApiField.TAG_ID: tag_meta_id, ApiField.IMAGE_ID: image_id, ApiField.ID: tag_id}
        resp = self._api.post('image-tags.remove-from-image', data)
        return resp.json()
