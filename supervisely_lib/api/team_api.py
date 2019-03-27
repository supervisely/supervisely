# coding: utf-8

from collections import namedtuple
from supervisely_lib.api.module_api import ApiField, ModuleNoParent
from supervisely_lib._utils import camel_to_snake


class TeamApi(ModuleNoParent):
    _info_sequence = [ApiField.ID,
                      ApiField.NAME,
                      ApiField.DESCRIPTION,
                      ApiField.ROLE,
                      ApiField.CREATED_AT,
                      ApiField.UPDATED_AT]
    Info = namedtuple('TeamInfo', [camel_to_snake(name) for name in _info_sequence])

    def get_list(self, filters=None):
        return self.get_list_all_pages('teams.list',  {ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'teams.info')

    def create(self, name, description=""):
        response = self.api.post('teams.add', {ApiField.NAME: name, ApiField.DESCRIPTION: description})
        return self._convert_json_info(response.json())

    def _get_update_method(self):
        return 'teams.editInfo'

