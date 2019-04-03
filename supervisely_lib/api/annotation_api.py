# coding: utf-8

from collections import namedtuple
import json

from supervisely_lib.api.module_api import ApiField, ModuleApi
from supervisely_lib._utils import camel_to_snake
from supervisely_lib._utils import batched


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

    def download_batch(self, dataset_id, image_ids, progress_cb=None):
        filters = [{"field": ApiField.IMAGE_ID, "operator": "in", "value": image_ids}]
        return self.get_list_all_pages('annotations.list', {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters}, progress_cb)

    # @TODO: no errors from api if annotation is not valid
    def upload(self, image_id: int, ann: dict):
        self.api.post('annotations.add', data={ApiField.IMAGE_ID: image_id, ApiField.ANNOTATION: ann})

    def upload_batch_paths(self, dataset_id, img_ids, ann_paths, progress_cb=None):
        MAX_BATCH_SIZE = 50
        for batch in batched(list(zip(img_ids, ann_paths)), MAX_BATCH_SIZE):
            data = []
            for img_id, ann_path in batch:
                with open(ann_path) as json_file:
                    ann_json = json.load(json_file)
                data.append({ApiField.IMAGE_ID: img_id, ApiField.ANNOTATION: ann_json})
            self.api.post('annotations.bulk.add', data={ApiField.DATASET_ID: dataset_id, ApiField.ANNOTATIONS: data})
            if progress_cb is not None:
                progress_cb(len(batch))

    def upload_batch(self, img_ids, anns, progress_cb=None):
        raise NotImplementedError()

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