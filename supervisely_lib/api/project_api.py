# coding: utf-8

from collections import namedtuple
from supervisely_lib.api.module_api import ApiField, ModuleApi
from supervisely_lib._utils import camel_to_snake


class ProjectApi(ModuleApi):
    _info_sequence = [ApiField.ID,
                      ApiField.NAME,
                      ApiField.DESCRIPTION,
                      ApiField.SIZE,
                      ApiField.README,
                      ApiField.WORKSPACE_ID,
                      ApiField.CREATED_AT,
                      ApiField.UPDATED_AT]
    Info = namedtuple('ProjectInfo', [camel_to_snake(name) for name in _info_sequence])

    def get_list(self, workspace_id, filters=None):
        return self.get_list_all_pages('projects.list',  {ApiField.WORKSPACE_ID: workspace_id, "filter": filters or []})

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'projects.info')

    def get_meta(self, id):
        response = self.api.post('projects.meta', {'id': id})
        return response.json()

    def create(self, workspace_id, name, description=""):
        response = self.api.post('projects.add', {ApiField.WORKSPACE_ID: workspace_id,
                                                  ApiField.NAME: name,
                                                  ApiField.DESCRIPTION: description})
        return self._convert_json_info(response.json())

    def _get_update_method(self):
        return 'projects.editInfo'

    def update_meta(self, id, meta):
        self.api.post('projects.meta.update', {ApiField.ID: id, ApiField.META: meta})

    def _clone_api_method_name(self):
        return 'projects.clone'

    def get_datasets_count(self, id):
        datasets = self.api.dataset.get_list(id)
        return len(datasets)

    def get_images_count(self, id):
        datasets = self.api.dataset.get_list(id)
        return sum([dataset.images_count for dataset in datasets])

