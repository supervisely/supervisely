# coding: utf-8

from supervisely_lib.api.module_api import ApiField, ModuleApi


class PluginApi(ModuleApi):
    @staticmethod
    def info_sequence():
        return [ApiField.ID,
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

    @staticmethod
    def info_tuple_name():
        return 'PluginInfo'

    def get_list(self, team_id, filters=None):
        '''
        :param team_id: int
        :param filters: list
        :return: list of plugins from the given team
        '''
        return self.get_list_all_pages('plugins.list',  {ApiField.TEAM_ID: team_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, team_id, plugin_id):
        '''
        :param team_id: int
        :param plugin_id: int
        :return: information about plugin from given plugin id in given team
        '''
        filters = [{"field": ApiField.ID, "operator": "=", "value": plugin_id}]
        return self._get_info_by_filters(team_id, filters)
