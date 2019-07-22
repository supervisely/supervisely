# coding: utf-8

from supervisely_lib.api.module_api import ApiField, CloneableModuleApi, UpdateableModule, RemoveableModuleApi
from supervisely_lib.project.project_meta import ProjectMeta


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
                ApiField.UPDATED_AT]

    @staticmethod
    def info_tuple_name():
        return 'ProjectInfo'

    def __init__(self, api):
        CloneableModuleApi.__init__(self, api)
        UpdateableModule.__init__(self, api)

    def get_list(self, workspace_id, filters=None):
        return self.get_list_all_pages('projects.list',  {ApiField.WORKSPACE_ID: workspace_id, "filter": filters or []})

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'projects.info')

    def get_meta(self, id):
        response = self._api.post('projects.meta', {'id': id})
        return response.json()

    def create(self, workspace_id, name, description="", change_name_if_conflict=False):
        effective_name = self._get_effective_new_name(
            parent_id=workspace_id, name=name, change_name_if_conflict=change_name_if_conflict)
        response = self._api.post('projects.add', {ApiField.WORKSPACE_ID: workspace_id,
                                                   ApiField.NAME: effective_name,
                                                   ApiField.DESCRIPTION: description})
        return self._convert_json_info(response.json())

    def _get_update_method(self):
        return 'projects.editInfo'

    def update_meta(self, id, meta):
        self._api.post('projects.meta.update', {ApiField.ID: id, ApiField.META: meta})

    def _clone_api_method_name(self):
        return 'projects.clone'

    def get_datasets_count(self, id):
        datasets = self._api.dataset.get_list(id)
        return len(datasets)

    def get_images_count(self, id):
        datasets = self._api.dataset.get_list(id)
        return sum([dataset.images_count for dataset in datasets])

    def _remove_api_method_name(self):
        return 'projects.remove'

    def merge_metas(self, src_project_id, dst_project_id):
        if src_project_id == dst_project_id:
            return self.get_meta(src_project_id)

        src_meta = ProjectMeta.from_json(self.get_meta(src_project_id))
        dst_meta = ProjectMeta.from_json(self.get_meta(dst_project_id))

        new_dst_meta = src_meta.merge(dst_meta)
        new_dst_meta_json = new_dst_meta.to_json()
        self.update_meta(dst_project_id, new_dst_meta.to_json())

        return new_dst_meta_json

