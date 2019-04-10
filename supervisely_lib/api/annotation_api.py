# coding: utf-8

from collections import namedtuple
import json

from supervisely_lib.api.module_api import ApiField, ModuleApi
from supervisely_lib._utils import camel_to_snake
from supervisely_lib._utils import batched
from supervisely_lib.annotation.annotation import Annotation


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
        id_to_ann = {}
        for batch in batched(image_ids):
            results = self.api.post('annotations.bulk.info', data={ApiField.DATASET_ID: dataset_id, ApiField.IMAGE_IDS: batch}).json()
            for ann_dict in results:
                ann_info = self._convert_json_info(ann_dict)
                id_to_ann[ann_info.image_id] = ann_info
            if progress_cb is not None:
                progress_cb(len(batch))
        ordered_results = [id_to_ann[image_id] for image_id in image_ids]
        return ordered_results

    def upload_path(self, img_id, ann_path):
        self.upload_paths([img_id], [ann_path])

    def upload_paths(self, img_ids, ann_paths, progress_cb=None):
        # img_ids from the same dataset
        def read_json(ann_path):
            with open(ann_path) as json_file:
                return json.load(json_file)
        self._upload_batch(read_json, img_ids, ann_paths, progress_cb)

    def upload_json(self, img_id, ann_json):
        self.upload_jsons([img_id], [ann_json])

    def upload_jsons(self, img_ids, ann_jsons, progress_cb=None):
        # img_ids from the same dataset
        self._upload_batch(lambda x : x, img_ids, ann_jsons, progress_cb)

    def upload_ann(self, img_id, ann):
        self.upload_anns([img_id], [ann])

    def upload_anns(self, img_ids, anns, progress_cb=None):
        # img_ids from the same dataset
        self._upload_batch(Annotation.to_json, img_ids, anns, progress_cb)

    def _upload_batch(self, func_ann_to_json, img_ids, anns, progress_cb=None):
        # img_ids from the same dataset
        if len(img_ids) == 0:
            return
        if len(img_ids) != len(anns):
            raise RuntimeError("Can not match \"img_ids\" and \"anns\" lists, len(img_ids) != len(anns)")

        dataset_id = self.api.image.get_info_by_id(img_ids[0]).dataset_id
        for batch in batched(list(zip(img_ids, anns))):
            data = [{ApiField.IMAGE_ID: img_id, ApiField.ANNOTATION: func_ann_to_json(ann)} for img_id, ann in batch]
            self.api.post('annotations.bulk.add', data={ApiField.DATASET_ID: dataset_id, ApiField.ANNOTATIONS: data})
            if progress_cb is not None:
                progress_cb(len(batch))

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