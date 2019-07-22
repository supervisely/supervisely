# coding: utf-8
from collections import namedtuple

from supervisely_lib.api.module_api import ApiField, ModuleApiBase, _get_single_item


class UserApi(ModuleApiBase):
    Membership = namedtuple("Membership", ['id', 'name', 'role_id', 'role'])

    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.LOGIN,
                #ApiField.ROLE_ID,
                ApiField.NAME,
                ApiField.EMAIL,
                ApiField.LOGINS,
                ApiField.DISABLED,
                ApiField.LAST_LOGIN,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT]

    @staticmethod
    def info_tuple_name():
        return 'UserInfo'

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'users.info')

    def get_info_by_login(self, login):
        filters = [{"field": ApiField.LOGIN, "operator": "=", "value": login}]
        items = self.get_list(filters)
        return _get_single_item(items)

    def get_list(self, filters=None):
        return self.get_list_all_pages('users.list', {ApiField.FILTER: filters or []})

    def create(self, login, password, is_restricted=False, name="", email=""):
        response = self._api.post('users.add', {ApiField.LOGIN: login,
                                                ApiField.PASSWORD: password,
                                                ApiField.IS_RESTRICTED: is_restricted,
                                                ApiField.NAME: name,
                                                ApiField.EMAIL: email,
                                                })
        return self.get_info_by_id(response.json()[ApiField.USER_ID])

    def _set_disabled(self, id, disable):
        self._api.post('users.disable', {ApiField.ID: id, ApiField.DISABLE: disable})

    def disable(self, id):
        self._set_disabled(id, True)

    def enable(self, id):
        self._set_disabled(id, False)

    def get_token(self, login):
        raise NotImplementedError()

    def get_teams(self, id):
        response = self._api.post('users.info', {ApiField.ID: id})
        teams_json = response.json()[ApiField.TEAMS]
        teams = []
        for team in teams_json:
            member = self.Membership(id=team[ApiField.ID],
                                     name=team[ApiField.NAME],
                                     role_id=team[ApiField.ROLE_ID],
                                     role=team[ApiField.ROLE])
            teams.append(member)
        return teams

    def add_to_team(self, user_id, team_id, role_id):
        user = self.get_info_by_id(user_id)
        response = self._api.post('members.add', {ApiField.LOGIN: user.login,
                                                  ApiField.TEAM_ID: team_id,
                                                  ApiField.ROLE_ID: role_id})

    def remove_from_team(self, user_id, team_id):
        response = self._api.post('members.remove', {ApiField.ID: user_id,
                                                     ApiField.TEAM_ID: team_id})

    def update(self, id, password=None, name=None):
        data = {}
        if password is not None:
            data[ApiField.PASSWORD] = password
        if name is not None:
            data[ApiField.NAME] = name
        if len(data) == 0:
            return
        data[ApiField.ID] = id

        self._api.post('users.editInfo', data)
        return self.get_info_by_id(id)

    def change_team_role(self, user_id, team_id, role_id):
        response = self._api.post('members.editInfo', {ApiField.ID: user_id,
                                                       ApiField.TEAM_ID: team_id,
                                                       ApiField.ROLE_ID: role_id})

    def get_team_members(self, team_id):
        team_members = self.get_list_all_pages('members.list', {ApiField.TEAM_ID: team_id, ApiField.FILTER: []},
                                               convert_json_info_cb=self._api.user._convert_json_info)
        return team_members

    def get_team_role(self, user_id, team_id):
        user_teams = self.get_teams(user_id)
        for member in user_teams:
            if member.id == team_id:
                return member
        return None
