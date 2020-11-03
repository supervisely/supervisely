# coding: utf-8

from enum import Enum
import pandas as pd
import urllib

from supervisely_lib.api.module_api import ApiField, CloneableModuleApi, UpdateableModule, RemoveableModuleApi
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.project.project_type import ProjectType


class ProjectNotFound(Exception):
    pass


class ExpectedProjectTypeMismatch(Exception):
    pass


class ProjectApi(CloneableModuleApi, UpdateableModule, RemoveableModuleApi):
    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.NAME,
                ApiField.DESCRIPTION,
                ApiField.SIZE,
                ApiField.README,
                ApiField.WORKSPACE_ID,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT,
                ApiField.TYPE,
                ApiField.REFERENCE_IMAGE_URL]

    @staticmethod
    def info_tuple_name():
        return 'ProjectInfo'

    def __init__(self, api):
        CloneableModuleApi.__init__(self, api)
        UpdateableModule.__init__(self, api)

    def get_list(self, workspace_id, filters=None):
        '''
        :param workspace_id: int
        :param filters: list
        :return: list all the projects for a given workspace
        '''
        return self.get_list_all_pages('projects.list',  {ApiField.WORKSPACE_ID: workspace_id, "filter": filters or []})

    def get_info_by_id(self, id, expected_type=None, raise_error=False):
        '''
        :param id: int
        :param expected_type: type of data we expext to get info (raise error if type of project is not None and != expected type)
        :param raise_error: bool
        :return: project metadata by numeric id (None if request status_code == 404, raise error in over way)
        '''
        info = self._get_info_by_id(id, 'projects.info')
        self._check_project_info(info, id=id, expected_type=expected_type, raise_error=raise_error)
        return info

    def get_info_by_name(self, parent_id, name, expected_type=None, raise_error=False):
        '''
        :param parent_id: int
        :param name: str
        :param expected_type: type of data we expext to get info
        :param raise_error: bool
        :return: project metadata by numeric workspace id and given name of project
        '''
        info = super().get_info_by_name(parent_id, name)
        self._check_project_info(info, name=name, expected_type=expected_type, raise_error=raise_error)
        return info

    def _check_project_info(self, info, id=None, name=None, expected_type=None, raise_error=False):
        '''
        Checks if a project exists with a given id and type of project == expected type
        :param info: project metadata information
        :param id: int
        :param name: str
        :param expected_type: type of data we expext to get info
        :param raise_error: bool
        '''
        if raise_error is False:
            return

        str_id = ""
        if id is not None:
            str_id += "id: {!r} ".format(id)
        if name is not None:
            str_id += "name: {!r}".format(name)

        if info is None:
            raise ProjectNotFound("Project {} not found".format(str_id))
        if expected_type is not None and info.type != str(expected_type):
            raise ExpectedProjectTypeMismatch("Project {!r} has type {!r}, but expected type is {!r}"
                                              .format(str_id, info.type, expected_type))

    def get_meta(self, id):
        '''
        :param id: int
        :return: labeling meta information for the project - the set of available object classes and tags
        '''
        response = self._api.post('projects.meta', {'id': id})
        return response.json()

    def create(self, workspace_id, name, type=ProjectType.IMAGES, description="", change_name_if_conflict=False):
        '''
        Create project with given name in workspace with given id
        :param workspace_id: int
        :param name: str
        :param type: type of progect to create
        :param description: str
        :param change_name_if_conflict: bool
        :return: created project metadata
        '''
        effective_name = self._get_effective_new_name(
            parent_id=workspace_id, name=name, change_name_if_conflict=change_name_if_conflict)
        response = self._api.post('projects.add', {ApiField.WORKSPACE_ID: workspace_id,
                                                   ApiField.NAME: effective_name,
                                                   ApiField.DESCRIPTION: description,
                                                   ApiField.TYPE: str(type)})
        return self._convert_json_info(response.json())

    def _get_update_method(self):
        return 'projects.editInfo'

    def update_meta(self, id, meta):
        '''
        Update given project with given metadata
        :param id: int
        :param meta: project metainformation
        '''
        self._api.post('projects.meta.update', {ApiField.ID: id, ApiField.META: meta})

    def _clone_api_method_name(self):
        return 'projects.clone'

    def get_datasets_count(self, id):
        '''
        :param id: int
        :return: int (number of datasets in given project)
        '''
        datasets = self._api.dataset.get_list(id)
        return len(datasets)

    def get_images_count(self, id):
        '''
        :param id: int
        :return: int (number of images in given project)
        '''
        datasets = self._api.dataset.get_list(id)
        return sum([dataset.images_count for dataset in datasets])

    def _remove_api_method_name(self):
        return 'projects.remove'

    def merge_metas(self, src_project_id, dst_project_id):
        '''
        Add metadata from given progect to given destination project
        :param src_project_id: int
        :param dst_project_id: int
        :return: merged project metainformation
        '''
        if src_project_id == dst_project_id:
            return self.get_meta(src_project_id)

        src_meta = ProjectMeta.from_json(self.get_meta(src_project_id))
        dst_meta = ProjectMeta.from_json(self.get_meta(dst_project_id))

        new_dst_meta = src_meta.merge(dst_meta)
        new_dst_meta_json = new_dst_meta.to_json()
        self.update_meta(dst_project_id, new_dst_meta.to_json())

        return new_dst_meta_json

    def get_activity(self, id):
        '''
        #@TODO - umar will add meta with review status and duration
        export const ACTIVITY_LOG = {
          LOGIN: 'login', - unavailable
          LOGOUT: 'logout', - unavailable
          CREATE_PROJECT: 'create_project',
          UPDATE_PROJECT: 'update_project',
          DISABLE_PROJECT: 'disable_project',
          RESTORE_PROJECT: 'restore_project',
          CREATE_DATASET: 'create_dataset',
          UPDATE_DATASET: 'update_dataset',
          DISABLE_DATASET: 'disable_dataset',
          RESTORE_DATASET: 'restore_dataset',
          CREATE_IMAGE: 'create_image',
          UPDATE_IMAGE: 'update_image',
          DISABLE_IMAGE: 'disable_image',
          RESTORE_IMAGE: 'restore_image',
          CREATE_FIGURE: 'create_figure',
          UPDATE_FIGURE: 'update_figure',
          DISABLE_FIGURE: 'disable_figure',
          RESTORE_FIGURE: 'restore_figure',
          CREATE_CLASS: 'create_class',
          UPDATE_CLASS: 'update_class',
          DISABLE_CLASS: 'disable_class',
          RESTORE_CLASS: 'restore_class',
          CREATE_BACKUP: 'create_backup',
          EXPORT_PROJECT: 'export_project',
          MODEL_TRAIN: 'model_train', - unavailable
          MODEL_INFERENCE: 'model_inference', - unavailable
          CREATE_PLUGIN: 'create_plugin', - unavailable
          DISABLE_PLUGIN: 'disable_plugin', - unavailable
          RESTORE_PLUGIN: 'restore_plugin', - unavailable
          CREATE_NODE: 'create_node', - unavailable
          DISABLE_NODE: 'disable_node',- unavailable
          RESTORE_NODE: 'restore_node',- unavailable
          CREATE_WORKSPACE: 'create_workspace',- unavailable
          DISABLE_WORKSPACE: 'disable_workspace',- unavailable
          RESTORE_WORKSPACE: 'restore_workspace',- unavailable
          CREATE_MODEL: 'create_model',- unavailable
          DISABLE_MODEL: 'disable_model',- unavailable
          RESTORE_MODEL: 'restore_model',- unavailable
          ADD_MEMBER: 'add_member',- unavailable
          REMOVE_MEMBER: 'remove_member',- unavailable
          LOGIN_TO_TEAM: 'login_to_team',
          ATTACH_TAG: 'attach_tag',
          UPDATE_TAG_VALUE: 'update_tag_value',
          DETACH_TAG: 'detach_tag',
          ANNOTATION_DURATION: 'annotation_duration',
          IMAGE_REVIEW_STATUS_UPDATED: 'image_review_status_updated',
        };

        '''
        response = self._api.post('projects.activity', {ApiField.ID: id})
        df = pd.DataFrame(response.json())
        return df

    def _convert_json_info(self, info: dict, skip_missing=True):
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        if res.reference_image_url is not None:
            res = res._replace(reference_image_url=urllib.parse.urljoin(self._api.server_address, res.reference_image_url))
        return res

    def get_stats(self, id):
        response = self._api.post('projects.stats', {ApiField.ID: id})
        return response.json()