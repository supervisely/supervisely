# coding: utf-8

from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.volume_annotation.volume_object_collection import VolumeObjectCollection
from supervisely_lib.api.entity_annotation.object_api import ObjectApi
from supervisely_lib.api.module_api import ApiField


class VolumeObjectApi(ObjectApi):
    def append_bulk(self, dataset_id, objects: VolumeObjectCollection, key_id_map: KeyIdMap = None):
        info = self._api.volume.get_info_by_id(dataset_id)
        return self._append_bulk(self._api.volume.tag, dataset_id, info.project_id, info.dataset_id, objects,
                                 key_id_map)

    def _get_volumetric_interpolation(self, volume_id, object_id):
        return self._api.post('figures.volumetric_interpolation', {ApiField.VOLUME_ID: volume_id,
                                                                   ApiField.OBJECT_ID: object_id})

    def get_volumetric_interpolation(self, volume_id, objects: VolumeObjectCollection, key_id_map):
        results = []
        for object in objects:
            object_id = key_id_map.get_object_id(object.key())
            response = self._get_volumetric_interpolation(volume_id, object_id)
            if response.status_code == 200:
                results.append(response.content)
            else:
                results.append(None)
        return results
