# coding: utf-8

from supervisely_lib.api.module_api import ApiField, ModuleApi, UpdateableModule


class WorkspaceApi(ModuleApi, UpdateableModule):
    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.NAME,
                ApiField.DESCRIPTION,
                ApiField.TEAM_ID,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT]

    @staticmethod
    def info_tuple_name():
        return 'WorkspaceInfo'

    def __init__(self, api):
        ModuleApi.__init__(self, api)
        UpdateableModule.__init__(self, api)

    def get_list(self, team_id, filters=None):
        '''
        :param team_id: int
        :param filters: list
        :return: list of all the workspaces in the selected team
        '''
        return self.get_list_all_pages('workspaces.list',  {ApiField.TEAM_ID: team_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        '''
        :param id: int
        :return: workspace metadata
        '''
        return self._get_info_by_id(id, 'workspaces.info')

    def create(self, team_id, name, description="", change_name_if_conflict=False):
        '''
        Create workspaces in team with given id
        :param team_id: int
        :param name: str
        :param description: str
        :param change_name_if_conflict: bool
        :return: new workspace metadata
        '''
        effective_name = self._get_effective_new_name(
            parent_id=team_id, name=name, change_name_if_conflict=change_name_if_conflict)
        response = self._api.post('workspaces.add', {ApiField.TEAM_ID: team_id,
                                                     ApiField.NAME: effective_name,
                                                     ApiField.DESCRIPTION: description})
        return self._convert_json_info(response.json())

    def _get_update_method(self):
        return 'workspaces.editInfo'
