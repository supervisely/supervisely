# coding: utf-8
from enum import IntEnum
from supervisely_lib.api.module_api import ApiField, ModuleApiBase


class RoleApi(ModuleApiBase):
    class DefaultRole(IntEnum):
        ADMIN =     1
        DEVELOPER = 2
        ANNOTATOR = 3
        VIEWER =    4

    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.ROLE,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT]

    @staticmethod
    def info_tuple_name():
        return 'RoleInfo'

    def get_list(self, filters=None):
        '''
        Print all roles that are available on private Supervisely instance
        :param filters: list
        :return: list
        '''
        return self.get_list_all_pages('roles.list', {ApiField.FILTER: filters or []})
