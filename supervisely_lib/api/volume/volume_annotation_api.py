# coding: utf-8

import json
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.video_annotation.video_annotation import VideoAnnotation

from supervisely_lib.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely_lib.io.json import load_json_file


class VolumeAnnotationApi(EntityAnnotationAPI):
    _method_download_bulk = 'volumes.annotations.bulk.info'
    _entity_ids_str = ApiField.VOLUME_IDS

    def download_bulk(self, dataset_id, volume_ids):
        response = self._api.post(self._method_download_bulk, {ApiField.DATASET_ID: dataset_id,
                                                               self._entity_ids_str: volume_ids})
        return response.json()

    def download(self, entity_id):
        dataset_id = self._api.volume.get_info_by_id(entity_id).dataset_id
        return self.download_bulk(dataset_id, [entity_id]).pop()

    def _append(self, tag_api, object_api, figure_api, project_id, dataset_id, entity_id, tags, objects, figures, key_id_map: KeyIdMap = None):
        raise NotImplementedError()

    def append(self, entity_id, ann, key_id_map: KeyIdMap = None):
        raise NotImplementedError()
