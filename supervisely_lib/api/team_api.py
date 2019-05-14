# coding: utf-8

from supervisely_lib.api.module_api import ApiField, ModuleNoParent, UpdateableModule


class TeamApi(ModuleNoParent, UpdateableModule):
    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.NAME,
                ApiField.DESCRIPTION,
                ApiField.ROLE,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT]

    @staticmethod
    def info_tuple_name():
        return 'TeamInfo'

    def __init__(self, api):
        ModuleNoParent.__init__(self, api)
        UpdateableModule.__init__(self, api)

    def get_list(self, filters=None):
        return self.get_list_all_pages('teams.list',  {ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'teams.info')

    def create(self, name, description="", change_name_if_conflict=False):
        effective_name = self._get_effective_new_name(name=name, change_name_if_conflict=change_name_if_conflict)
        response = self._api.post('teams.add', {ApiField.NAME: effective_name, ApiField.DESCRIPTION: description})
        return self._convert_json_info(response.json())

    def _get_update_method(self):
        return 'teams.editInfo'

