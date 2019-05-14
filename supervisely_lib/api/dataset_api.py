# coding: utf-8

from supervisely_lib.api.module_api import ApiField, ModuleApi, UpdateableModule


class DatasetApi(ModuleApi, UpdateableModule):
    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.NAME,
                ApiField.DESCRIPTION,
                ApiField.SIZE,
                ApiField.PROJECT_ID,
                ApiField.IMAGES_COUNT,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT]

    @staticmethod
    def info_tuple_name():
        return 'DatasetInfo'

    def __init__(self, api):
        ModuleApi.__init__(self, api)
        UpdateableModule.__init__(self, api)

    def get_list(self, project_id, filters=None):
        return self.get_list_all_pages('datasets.list',  {ApiField.PROJECT_ID: project_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'datasets.info')

    def create(self, project_id, name, description="", change_name_if_conflict=False):
        effective_name = self._get_effective_new_name(
            parent_id=project_id, name=name, change_name_if_conflict=change_name_if_conflict)
        response = self._api.post('datasets.add', {ApiField.PROJECT_ID: project_id,
                                                   ApiField.NAME: effective_name,
                                                   ApiField.DESCRIPTION: description})
        return self._convert_json_info(response.json())

    def _get_update_method(self):
        return 'datasets.editInfo'
