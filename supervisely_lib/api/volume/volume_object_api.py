# coding: utf-8

from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.volume_annotation.volume_object_collection import VolumeObjectCollection
from supervisely_lib.api.entity_annotation.object_api import ObjectApi


class VolumeObjectApi(ObjectApi):
    def append_bulk(self, dataset_id, objects: VolumeObjectCollection, key_id_map: KeyIdMap = None):
        info = self._api.volume.get_info_by_id(dataset_id)
        return self._append_bulk(self._api.volume.tag, dataset_id, info.project_id, info.dataset_id, objects,
                                 key_id_map)
