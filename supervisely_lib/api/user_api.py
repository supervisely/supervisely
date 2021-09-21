# coding: utf-8
from collections import namedtuple
import pandas as pd

from supervisely_lib.api.module_api import ApiField, ModuleApiBase, _get_single_item


class UserApi(ModuleApiBase):
    Membership = namedtuple("Membership", ['id', 'name', 'role_id', 'role'])

    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.LOGIN,
                ApiField.ROLE,
                ApiField.ROLE_ID,
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

    def _convert_json_info(self, info: dict, skip_missing=True):
        return super(UserApi, self)._convert_json_info(info, skip_missing=skip_missing)

    def get_info_by_id(self, id):
        '''
        :param id: int
        :return: user metainformation with given id
        '''
        return self._get_info_by_id(id, 'users.info')

    def get_info_by_login(self, login):
        '''
        :param login: str
        :return: user metainformation with given login
        '''
        filters = [{"field": ApiField.LOGIN, "operator": "=", "value": login}]
        items = self.get_list(filters)
        return _get_single_item(items)

    def get_member_info_by_login(self, team_id, login):
        filters = [{"field": ApiField.LOGIN, "operator": "=", "value": login}]
        team_members = self.get_list_all_pages('members.list', {ApiField.TEAM_ID: team_id, ApiField.FILTER: filters},
                                               convert_json_info_cb=self._api.user._convert_json_info)
        return _get_single_item(team_members)

    def get_member_info_by_id(self, team_id, user_id):
        filters = [{"field": ApiField.ID, "operator": "=", "value": user_id}]
        team_members = self.get_list_all_pages('members.list', {ApiField.TEAM_ID: team_id, ApiField.FILTER: filters},
                                               convert_json_info_cb=self._api.user._convert_json_info)
        return _get_single_item(team_members)

    def get_list(self, filters=None):
        '''
        :param filters: list
        :return: list of all registered users
        '''
        return self.get_list_all_pages('users.list', {ApiField.FILTER: filters or []})

    def create(self, login, password, is_restricted=False, name="", email=""):
        '''
        Create new user with given login and password
        :param login: str
        :param password: str
        :param is_restricted: bool
        :param name: str
        :param email: str
        :return: new user metainformation
        '''
        response = self._api.post('users.add', {ApiField.LOGIN: login,
                                                ApiField.PASSWORD: password,
                                                ApiField.IS_RESTRICTED: is_restricted,
                                                ApiField.NAME: name,
                                                ApiField.EMAIL: email,
                                                })
        return self.get_info_by_id(response.json()[ApiField.USER_ID])

    def _set_disabled(self, id, disable):
        '''
        Check status of the user with given id
        :param id: int
        :param disable: bool
        '''
        self._api.post('users.disable', {ApiField.ID: id, ApiField.DISABLE: disable})

    def disable(self, id):
        '''
        Check user with given id is disable
        :param id: int
        '''
        self._set_disabled(id, True)

    def enable(self, id):
        '''
        Check user with given id is enable
        :param id: int
        '''
        self._set_disabled(id, False)

    def get_token(self, login):
        raise NotImplementedError()

    def get_teams(self, id):
        '''
        :param id: int
        :return: list of teams in which given user is a member of
        '''
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
        '''
        Invite user to team
        :param user_id: int
        :param team_id: int
        :param role_id: int
        '''
        user = self.get_info_by_id(user_id)
        response = self._api.post('members.add', {ApiField.LOGIN: user.login,
                                                  ApiField.TEAM_ID: team_id,
                                                  ApiField.ROLE_ID: role_id})

    def remove_from_team(self, user_id, team_id):
        '''
        Remove user from team
        :param user_id: int
        :param team_id: int
        '''
        response = self._api.post('members.remove', {ApiField.ID: user_id,
                                                     ApiField.TEAM_ID: team_id})

    def update(self, id, password=None, name=None):
        '''
        Update user info
        :param id: int
        :param password: str
        :param name: str
        :return: updated user metainformation
        '''
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
        '''
        Change user role in team
        :param user_id: int
        :param team_id: int
        :param role_id: int
        '''
        response = self._api.post('members.editInfo', {ApiField.ID: user_id,
                                                       ApiField.TEAM_ID: team_id,
                                                       ApiField.ROLE_ID: role_id})

    def get_team_members(self, team_id):
        '''
        :param team_id: int
        :return: list all team users with corresponding roles
        '''
        team_members = self.get_list_all_pages('members.list', {ApiField.TEAM_ID: team_id, ApiField.FILTER: []},
                                               convert_json_info_cb=self._api.user._convert_json_info)
        return team_members

    def get_team_role(self, user_id, team_id):
        '''
        Checks if user with given belongs to team with given id
        :param user_id: int
        :param team_id: int
        '''
        user_teams = self.get_teams(user_id)
        for member in user_teams:
            if member.id == team_id:
                return member
        return None

    def get_member_activity(self, team_id, user_id, progress_cb=None):
        '''
        :param team_id: int
        :param user_id: int
        :param progress_cb: fn
        :return: pandas dataframe (table with activity data of given user)
        '''
        activity = self._api.team.get_activity(team_id, filter_user_id=user_id, progress_cb=progress_cb)
        df = pd.DataFrame(activity)
        return df

    def add_to_team_by_login(self, user_login, team_id, role_id):
        response = self._api.post('members.add', {ApiField.LOGIN: user_login,
                                                  ApiField.TEAM_ID: team_id,
                                                  ApiField.ROLE_ID: role_id})
        return response.json()