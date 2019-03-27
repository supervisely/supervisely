# coding: utf-8

from collections import namedtuple
import json

from supervisely_lib.api.module_api import ApiField, ModuleApi
from supervisely_lib._utils import camel_to_snake


class AnnotationApi(ModuleApi):
    _info_sequence = [ApiField.IMAGE_ID,
                      ApiField.IMAGE_NAME,
                      ApiField.ANNOTATION,
                      ApiField.CREATED_AT,
                      ApiField.UPDATED_AT]
    Info = namedtuple('AnnotationInfo', [camel_to_snake(name) for name in _info_sequence])

    def get_list(self, dataset_id, filters=None, progress_cb=None):
        return self.get_list_all_pages('annotations.list',  {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters or []}, progress_cb)

    def download(self, image_id):
        response = self.api.post('annotations.info', {ApiField.IMAGE_ID: image_id})
        return self._convert_json_info(response.json())

    # @TODO: no errors from api if annotation is not valid
    def upload(self, image_id: int, ann: dict):
        self.api.post('annotations.add', data={ApiField.IMAGE_ID: image_id, ApiField.ANNOTATION: ann})

    def add_batch(self, img_ids, ann_paths, progress_cb=None):
        for ann_path, img_id in zip(ann_paths, img_ids):
            ann = json.load(open(ann_path, 'r'))
            self.upload(img_id, ann)
            if progress_cb is not None:
                progress_cb()

    def get_info_by_id(self, id):
        raise RuntimeError('Method is not supported')

    def get_info_by_name(self, parent_id, name):
        raise RuntimeError('Method is not supported')

    def exists(self, parent_id, name):
        raise RuntimeError('Method is not supported')

    def get_free_name(self, parent_id, name):
        raise RuntimeError('Method is not supported')

    def _add_sort_param(self, data):
        return data