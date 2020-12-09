# coding: utf-8

import urllib
from supervisely_lib.api.module_api import ApiField, ModuleApi, UpdateableModule, RemoveableModuleApi


class DatasetApi(UpdateableModule, RemoveableModuleApi):
    '''
    This is a class for creating and using DatasetApi objects. Here is where your labeled and unlabeled images and other
    files live. There is no more levels: images or videos are directly attached to a dataset. Dataset is a unit of work.
    All images or videos are directly attached to a dataset. A dataset is some sort of data folder with stuff to annotate.
    '''
    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.NAME,
                ApiField.DESCRIPTION,
                ApiField.SIZE,
                ApiField.PROJECT_ID,
                ApiField.IMAGES_COUNT,
                ApiField.ITEMS_COUNT,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT,
                ApiField.REFERENCE_IMAGE_URL]

    @staticmethod
    def info_tuple_name():
        return 'DatasetInfo'

    def __init__(self, api):
        ModuleApi.__init__(self, api)
        UpdateableModule.__init__(self, api)

    def get_list(self, project_id, filters=None):
        '''
        :param project_id: int
        :param filters: list
        :return: list all the datasets for a given project
        '''
        return self.get_list_all_pages('datasets.list',  {ApiField.PROJECT_ID: project_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        '''
        :param id: int
        :return: dataset metadata by numeric id
        '''
        return self._get_info_by_id(id, 'datasets.info')

    def create(self, project_id, name, description="", change_name_if_conflict=False):
        '''
        Create dataset with given name in project with given id
        :param project_id: int
        :param name: str
        :param description: str
        :param change_name_if_conflict: bool
        :return: created project dataset metadata
        '''
        effective_name = self._get_effective_new_name(
            parent_id=project_id, name=name, change_name_if_conflict=change_name_if_conflict)
        response = self._api.post('datasets.add', {ApiField.PROJECT_ID: project_id,
                                                   ApiField.NAME: effective_name,
                                                   ApiField.DESCRIPTION: description})
        return self._convert_json_info(response.json())

    def get_or_create(self, project_id, name, description=""):
        '''
        Check if dataset with given name exist in project, if not it create dataset with given name in project
        :param project_id: int
        :param name: str
        :param description: str
        :return: dataset metadata
        '''
        dataset_info = self.get_info_by_name(project_id, name)
        if dataset_info is None:
            dataset_info = self.create(project_id, name, description=description)
        return dataset_info

    def _get_update_method(self):
        return 'datasets.editInfo'

    def _remove_api_method_name(self):
        return 'datasets.remove'

    def copy_batch(self, dst_project_id, ids, new_names=None, change_name_if_conflict=False, with_annotations=False):
        '''
        Copy given datasets in destination project
        :param dst_project_id: int
        :param ids: list of integers
        :param new_names: list of str (if new_names not None and lengh of ids list != lengh of new_names list generate error)
        :param change_name_if_conflict: bool
        :param with_annotations: bool
        :return: list of datasets (DatasetApi class objects)
        '''
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
        '''
        Copy given dataset in destination project
        :param dst_project_id: int
        :param id: int
        :param new_name: str
        :param change_name_if_conflict: bool
        :param with_annotations: bool
        :return: DatasetApi class object
        '''
        new_datasets = self.copy_batch(dst_project_id, [id], [new_name], change_name_if_conflict, with_annotations)
        if len(new_datasets) == 0:
            return None
        return new_datasets[0]

    def move_batch(self, dst_project_id, ids, new_names, change_name_if_conflict=False, with_annotations=False):
        '''
        Moves given datasets in destination project
        :param dst_project_id: int
        :param ids: list of integers
        :param new_names: list of str (if new_names not None and lengh of ids list != lengh of new_names list generate error)
        :param change_name_if_conflict: bool
        :param with_annotations: bool
        :return: list of datasets (DatasetApi class objects)
        '''
        new_datasets = self.copy_batch(dst_project_id, ids, new_names, change_name_if_conflict, with_annotations)
        self.remove_batch(ids)
        return new_datasets

    def move(self, dst_project_id, id, new_name, change_name_if_conflict=False, with_annotations=False):
        '''
        Move given dataset in destination project
        :param dst_project_id: int
        :param id: int
        :param new_name: str
        :param change_name_if_conflict: bool
        :param with_annotations: bool
        :return: DatasetApi class object
        '''
        new_dataset = self.copy(dst_project_id, id, new_name, change_name_if_conflict, with_annotations)
        self.remove(id)
        return new_dataset

    def _convert_json_info(self, info: dict, skip_missing=True):
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        if res.reference_image_url is not None:
            res = res._replace(reference_image_url=urllib.parse.urljoin(self._api.server_address, res.reference_image_url))
        if res.items_count is None:
            res = res._replace(items_count=res.images_count)
        return res

