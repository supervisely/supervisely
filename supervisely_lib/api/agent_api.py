# coding: utf-8

from enum import Enum
from supervisely_lib.api.module_api import ApiField, ModuleApi, ModuleWithStatus


class AgentNotFound(Exception):
    pass


class AgentNotRunning(Exception):
    pass


class AgentApi(ModuleApi, ModuleWithStatus):
    class Status(Enum):
        WAITING = 'waiting'
        RUNNING = 'running'

    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.NAME,
                ApiField.TOKEN,
                ApiField.STATUS,
                ApiField.USER_ID,
                ApiField.TEAM_ID,
                ApiField.CAPABILITIES,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT]

    @staticmethod
    def info_tuple_name():
        return 'AgentInfo'

    def __init__(self, api):
        ModuleApi.__init__(self, api)
        ModuleWithStatus.__init__(self)

    def get_list(self, team_id, filters=None):
        return self.get_list_all_pages('agents.list',  {'teamId': team_id, "filter": filters or []})

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'agent.info')

    def get_status(self, id):
        status_str = self.get_info_by_id(id).status
        return self.Status(status_str)

    def raise_for_status(self, status):
        pass
