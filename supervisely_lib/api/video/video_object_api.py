# coding: utf-8

from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.video_annotation.video_object_collection import VideoObjectCollection
from supervisely_lib.api.entity_annotation.object_api import ObjectApi


class VideoObjectApi(ObjectApi):
    def append_bulk(self, video_id, objects: VideoObjectCollection, key_id_map: KeyIdMap = None):
        info = self._api.video.get_info_by_id(video_id)
        return self._append_bulk(self._api.video.tag, video_id, info.project_id, info.dataset_id, objects, key_id_map)
