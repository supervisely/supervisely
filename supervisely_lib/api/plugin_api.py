# coding: utf-8

from collections import namedtuple
from supervisely_lib.api.module_api import ApiField, ModuleApi
from supervisely_lib._utils import camel_to_snake


class PluginApi(ModuleApi):
    _info_sequence = [ApiField.ID,
                      ApiField.NAME,
                      ApiField.DESCRIPTION,
                      ApiField.TYPE,
                      ApiField.DEFAULT_VERSION,
                      ApiField.DOCKER_IMAGE,
                      ApiField.README,
                      ApiField.CONFIGS,
                      ApiField.VERSIONS,
                      ApiField.CREATED_AT,
                      ApiField.UPDATED_AT]
    Info = namedtuple('PluginInfo', [camel_to_snake(name) for name in _info_sequence])

    def get_list(self, team_id, filters=None):
        return self.get_list_all_pages('plugins.list',  {ApiField.TEAM_ID: team_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, team_id, plugin_id):
        filters = [{"field": ApiField.ID, "operator": "=", "value": plugin_id}]
        return self._get_info_by_filters(team_id, filters)
