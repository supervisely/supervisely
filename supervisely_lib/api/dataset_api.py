# coding: utf-8

from supervisely_lib.api.module_api import ApiField, ModuleApi, UpdateableModule, RemoveableModuleApi


class DatasetApi(UpdateableModule, RemoveableModuleApi):
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

    def _remove_api_method_name(self):
        return 'datasets.remove'

    def copy_batch(self, dst_project_id, ids, new_names=None, change_name_if_conflict=False, with_annotations=False):
        if new_names is not None and len(ids) != len(new_names):
            raise RuntimeError('Can not match "ids" and "new_names" lists, len(ids) != len(new_names)')

        new_datasets = []
        for idx, dataset_id in enumerate(ids):
            dataset = self.get_info_by_id(dataset_id)
            new_dataset_name = dataset.name
            if new_names is not None:
                new_dataset_name = new_names[idx]
            src_images = self._api.image.get_list(dataset.id)
            src_image_ids = [image.id for image in src_images]
            new_dataset = self._api.dataset.create(dst_project_id, new_dataset_name, dataset.description,
                                                   change_name_if_conflict=change_name_if_conflict)
            self._api.image.copy_batch(new_dataset.id, src_image_ids, change_name_if_conflict, with_annotations)
            new_datasets.append(new_dataset)
        return new_datasets

    def copy(self, dst_project_id, id, new_name=None, change_name_if_conflict=False, with_annotations=False):
        new_datasets = self.copy_batch(dst_project_id, [id], [new_name], change_name_if_conflict, with_annotations)
        if len(new_datasets) == 0:
            return None
        return new_datasets[0]

    def move_batch(self, dst_project_id, ids, new_names, change_name_if_conflict=False, with_annotations=False):
        new_datasets = self.copy_batch(dst_project_id, ids, new_names, change_name_if_conflict, with_annotations)
        self.remove_batch(ids)
        return new_datasets

    def move(self, dst_project_id, id, new_name, change_name_if_conflict=False, with_annotations=False):
        new_dataset = self.copy(dst_project_id, id, new_name, change_name_if_conflict, with_annotations)
        self.remove(id)
        return new_dataset

