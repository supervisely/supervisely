# coding: utf-8

from supervisely_lib.api.module_api import ModuleApi
from supervisely_lib.api.module_api import ApiField


class ProjectClassApi(ModuleApi):
    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                #ApiField.PROJECT_ID,
                ApiField.NAME,
                ApiField.DESCRIPTION,
                ApiField.SHAPE,
                ApiField.COLOR,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT
                ]

    @staticmethod
    def info_tuple_name():
        return 'ProjectClassInfo'

    def get_list(self, project_id, filters=None):
        '''
        :param project_id: int
        :param filters: list
        :return: list all the classes for a given project
        '''
        return self.get_list_all_pages('advanced.object_classes.list',  {ApiField.PROJECT_ID: project_id, "filter": filters or []})
