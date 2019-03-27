# coding: utf-8

from collections import namedtuple
from supervisely_lib.api.module_api import ApiField, ModuleApi
from supervisely_lib._utils import camel_to_snake


class DatasetApi(ModuleApi):
    _info_sequence = [ApiField.ID,
                      ApiField.NAME,
                      ApiField.DESCRIPTION,
                      ApiField.SIZE,
                      ApiField.PROJECT_ID,
                      ApiField.IMAGES_COUNT,
                      ApiField.CREATED_AT,
                      ApiField.UPDATED_AT]
    Info = namedtuple('DatasetInfo', [camel_to_snake(name) for name in _info_sequence])

    def get_list(self, project_id, filters=None):
        return self.get_list_all_pages('datasets.list',  {ApiField.PROJECT_ID: project_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'datasets.info')

    def create(self, project_id, name, description=""):
        response = self.api.post('datasets.add', {ApiField.PROJECT_ID: project_id,
                                                  ApiField.NAME: name,
                                                  ApiField.DESCRIPTION: description})
        return self._convert_json_info(response.json())

    def _get_update_method(self):
        return 'datasets.editInfo'
