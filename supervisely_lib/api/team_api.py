# coding: utf-8

from supervisely_lib.api.module_api import ApiField, ModuleNoParent, UpdateableModule
from typing import List

#@TODO - umar will add meta with review status and duration
class ActivityAction:
    LOGIN = 'login'
    LOGOUT = 'logout'
    CREATE_PROJECT = 'create_project'
    UPDATE_PROJECT = 'update_project'
    DISABLE_PROJECT = 'disable_project'
    RESTORE_PROJECT = 'restore_project'
    CREATE_DATASET = 'create_dataset'
    UPDATE_DATASET = 'update_dataset'
    DISABLE_DATASET = 'disable_dataset'
    RESTORE_DATASET = 'restore_dataset'
    CREATE_IMAGE = 'create_image'
    UPDATE_IMAGE = 'update_image'
    DISABLE_IMAGE = 'disable_image'
    RESTORE_IMAGE = 'restore_image'
    CREATE_FIGURE = 'create_figure'
    UPDATE_FIGURE = 'update_figure'
    DISABLE_FIGURE = 'disable_figure'
    RESTORE_FIGURE = 'restore_figure'
    CREATE_CLASS = 'create_class'
    UPDATE_CLASS = 'update_class'
    DISABLE_CLASS = 'disable_class'
    RESTORE_CLASS = 'restore_class'
    CREATE_BACKUP = 'create_backup'
    EXPORT_PROJECT = 'export_project'
    MODEL_TRAIN = 'model_train'
    MODEL_INFERENCE = 'model_inference'
    CREATE_PLUGIN = 'create_plugin'
    DISABLE_PLUGIN = 'disable_plugin'
    RESTORE_PLUGIN = 'restore_plugin'
    CREATE_NODE = 'create_node'
    DISABLE_NODE = 'disable_node'
    RESTORE_NODE = 'restore_node'
    CREATE_WORKSPACE = 'create_workspace'
    DISABLE_WORKSPACE = 'disable_workspace'
    RESTORE_WORKSPACE = 'restore_workspace'
    CREATE_MODEL = 'create_model'
    DISABLE_MODEL = 'disable_model'
    RESTORE_MODEL = 'restore_model'
    ADD_MEMBER = 'add_member'
    REMOVE_MEMBER = 'remove_member'
    LOGIN_TO_TEAM = 'login_to_team'
    ATTACH_TAG = 'attach_tag'
    UPDATE_TAG_VALUE = 'update_tag_value'
    DETACH_TAG = 'detach_tag'
    ANNOTATION_DURATION = 'annotation_duration'
    IMAGE_REVIEW_STATUS_UPDATED = 'image_review_status_updated'

    # case #1 - labeler pressed "finish image" button in labeling job
    # action: IMAGE_REVIEW_STATUS_UPDATED -> meta["reviewStatus"] == 'done'

    # case #2 - reviewer pressed "accept" or "reject" button
    # action: IMAGE_REVIEW_STATUS_UPDATED -> meta["reviewStatus"] == 'accepted' or 'rejected'

    # possible review statuses:
    # 'done' - i.e. labeler finished the image,
    # 'accepted' - reviewer
    # 'rejected' - reviewer

    # case #3 duration
    # action: ANNOTATION_DURATION -> meta["duration"] e.g. meta-> {"duration": 30} in seconds


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
        '''
        :param filters: list
        :return: list of all teams
        '''
        return self.get_list_all_pages('teams.list',  {ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        '''
        :param id: int
        :return: team metainformation with given id
        '''
        return self._get_info_by_id(id, 'teams.info')

    def create(self, name, description="", change_name_if_conflict=False):
        '''
        Create team with given name
        :param name: str
        :param description: str
        :param change_name_if_conflict: bool
        :return: team metainformation
        '''
        effective_name = self._get_effective_new_name(name=name, change_name_if_conflict=change_name_if_conflict)
        response = self._api.post('teams.add', {ApiField.NAME: effective_name, ApiField.DESCRIPTION: description})
        return self._convert_json_info(response.json())

    def _get_update_method(self):
        return 'teams.editInfo'

    def get_activity(self, team_id,
                     filter_user_id=None, filter_project_id=None, filter_job_id=None, filter_actions=None):
        filters = []
        if filter_user_id is not None:
            filters.append({"field": ApiField.USER_ID, "operator": "=", "value": filter_user_id})
        if filter_project_id is not None:
            filters.append({"field": ApiField.PROJECT_ID, "operator": "=", "value": filter_project_id})
        if filter_job_id is not None:
            filters.append({"field": ApiField.JOB_ID, "operator": "=", "value": filter_job_id})
        if filter_actions is not None:
            if type(filter_actions) is not list:
                raise TypeError("type(filter_actions) is {!r}. But has to be of type {!r}".format(type(filter_actions), list))
            filters.append({"field": ApiField.TYPE, "operator": "in", "value": filter_actions})

        method = 'teams.activity'
        data = {ApiField.TEAM_ID: team_id, ApiField.FILTER: filters}
        first_response = self._api.post(method, data)
        first_response = first_response.json()

        total = first_response['total']
        per_page = first_response['perPage']
        pages_count = first_response['pagesCount']
        results = first_response['entities']

        if pages_count == 1 and len(first_response['entities']) == total:
            pass
        else:
            for page_idx in range(2, pages_count + 1):
                temp_resp = self._api.post(method, {**data, 'page': page_idx, 'per_page': per_page})
                temp_items = temp_resp.json()['entities']
                results.extend(temp_items)
            if len(results) != total:
                raise RuntimeError('Method {!r}: error during pagination, some items are missed'.format(method))

        return results
