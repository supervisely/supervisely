# coding: utf-8

import json
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.video_annotation.video_annotation import VideoAnnotation

from supervisely_lib.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely_lib.io.json import load_json_file


class VideoAnnotationAPI(EntityAnnotationAPI):
    _method_download_bulk = 'videos.annotations.bulk.info'
    _entity_ids_str = ApiField.VIDEO_IDS

    def download(self, video_id):
        '''
        :param video_id: int
        :return: video annotation to given id in json format
        '''
        video_info = self._api.video.get_info_by_id(video_id)
        return self._download(video_info.dataset_id, video_id)

    def append(self, video_id, ann: VideoAnnotation, key_id_map: KeyIdMap = None):
        '''
        ???
        :param video_id: int
        :param ann: VideoAnnotation class object
        :param key_id_map: KeyIdMap class object
        '''
        info = self._api.video.get_info_by_id(video_id)
        self._append(self._api.video.tag, self._api.video.object, self._api.video.figure,
                     info.project_id, info.dataset_id, video_id,
                     ann.tags, ann.objects, ann.figures, key_id_map)

    def upload_paths(self, video_ids, ann_paths, project_meta, progress_cb=None):
        # video_ids from the same dataset

        for video_id, ann_path in zip(video_ids, ann_paths):
            ann_json = load_json_file(ann_path)
            ann = VideoAnnotation.from_json(ann_json, project_meta)

            # ignore existing key_id_map because the new objects will be created
            self.append(video_id, ann)