# coding: utf-8

from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume_annotation.volume_object_collection import (
    VolumeObjectCollection,
)
from supervisely.api.entity_annotation.object_api import ObjectApi


class VolumeObjectApi(ObjectApi):
    def append_bulk(
        self, volume_id, objects: VolumeObjectCollection, key_id_map: KeyIdMap = None
    ):
        info = self._api.volume.get_info_by_id(volume_id)
        return self._append_bulk(
            self._api.volume.tag,
            volume_id,
            info.project_id,
            info.dataset_id,
            objects,
            key_id_map,
        )
