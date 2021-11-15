# coding: utf-8

from supervisely_lib.api.module_api import ApiField
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely_lib.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI


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

    def append(self, volume_id, ann: VolumeAnnotation, key_id_map: KeyIdMap = None):
        info = self._api.volume.get_info_by_id(volume_id)
        self._append(self._api.volume.tag, self._api.volume.object, self._api.volume.figure,
                     info.project_id, info.dataset_id, volume_id,
                     ann.tags, ann.objects, ann.figures, key_id_map)
