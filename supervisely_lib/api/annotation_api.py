# coding: utf-8

import json

from supervisely_lib.annotation.annotation import Annotation
from supervisely_lib.api.module_api import ApiField, ModuleApi
from supervisely_lib._utils import batched


class AnnotationApi(ModuleApi):
    @staticmethod
    def info_sequence():
        return [ApiField.IMAGE_ID,
                ApiField.IMAGE_NAME,
                ApiField.ANNOTATION,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT]

    @staticmethod
    def info_tuple_name():
        return 'AnnotationInfo'

    def get_list(self, dataset_id, filters=None, progress_cb=None):
        '''
        :param dataset_id: int
        :param filters: list
        :param progress_cb:
        :return: list all the annotations for a given dataset
        '''
        return self.get_list_all_pages('annotations.list',  {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters or []}, progress_cb)

    def download(self, image_id, with_custom_data=False):
        '''
        :param image_id: int
        :return: serialized JSON annotation for the image id
        '''
        response = self._api.post('annotations.info',
                                  {ApiField.IMAGE_ID: image_id, ApiField.WITH_CUSTOM_DATA: with_custom_data})
        return self._convert_json_info(response.json())

    def download_batch(self, dataset_id, image_ids, progress_cb=None, with_custom_data=False):
        '''
        :param dataset_id: int
        :param image_ids: list of integers
        :param progress_cb:
        :return: list of serialized JSON annotations for the given dataset id and image id's
        '''
        id_to_ann = {}
        for batch in batched(image_ids):
            post_data = {
                ApiField.DATASET_ID: dataset_id,
                ApiField.IMAGE_IDS: batch,
                ApiField.WITH_CUSTOM_DATA: with_custom_data
            }
            results = self._api.post('annotations.bulk.info', data=post_data).json()
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
        self._upload_batch(lambda x: x, img_ids, ann_jsons, progress_cb)

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
            raise RuntimeError('Can not match "img_ids" and "anns" lists, len(img_ids) != len(anns)')

        dataset_id = self._api.image.get_info_by_id(img_ids[0]).dataset_id
        for batch in batched(list(zip(img_ids, anns))):
            data = [{ApiField.IMAGE_ID: img_id, ApiField.ANNOTATION: func_ann_to_json(ann)} for img_id, ann in batch]
            self._api.post('annotations.bulk.add', data={ApiField.DATASET_ID: dataset_id, ApiField.ANNOTATIONS: data})
            if progress_cb is not None:
                progress_cb(len(batch))

    def get_info_by_id(self, id):
        raise NotImplementedError('Method is not supported')

    def get_info_by_name(self, parent_id, name):
        raise NotImplementedError('Method is not supported')

    def exists(self, parent_id, name):
        raise NotImplementedError('Method is not supported')

    def get_free_name(self, parent_id, name):
        raise NotImplementedError('Method is not supported')

    def _add_sort_param(self, data):
        return data

    def copy_batch(self, src_image_ids, dst_image_ids, progress_cb=None):
        if len(src_image_ids) != len(dst_image_ids):
            raise RuntimeError('Can not match "src_image_ids" and "dst_image_ids" lists, '
                               'len(src_image_ids) != len(dst_image_ids)')
        if len(src_image_ids) == 0:
            return

        src_dataset_id = self._api.image.get_info_by_id(src_image_ids[0]).dataset_id
        for cur_batch in batched(list(zip(src_image_ids, dst_image_ids))):
            src_ids_batch, dst_ids_batch = zip(*cur_batch)
            ann_infos = self.download_batch(src_dataset_id, src_ids_batch)
            ann_jsons = [ann_info.annotation for ann_info in ann_infos]
            self.upload_jsons(dst_ids_batch, ann_jsons)
            if progress_cb is not None:
                progress_cb(len(src_ids_batch))

    def copy(self, src_image_id, dst_image_id):
        self.copy_batch([src_image_id], [dst_image_id])

    def copy_batch_by_ids(self, src_image_ids, dst_image_ids):
        if len(src_image_ids) != len(dst_image_ids):
            raise RuntimeError('Can not match "src_image_ids" and "dst_image_ids" lists, '
                               'len(src_image_ids) != len(dst_image_ids)')
        if len(src_image_ids) == 0:
            return

        self._api.post('annotations.bulk.copy', data={"srcImageIds": src_image_ids,
                                                      "destImageIds": dst_image_ids,
                                                      "preserveSourceDate": True})
