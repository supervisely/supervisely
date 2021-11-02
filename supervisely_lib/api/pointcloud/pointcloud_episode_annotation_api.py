# coding: utf-8

from supervisely_lib.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.pointcloud_annotation.pointcloud_episode_annotation import PointcloudEpisodeAnnotation
from supervisely_lib.video_annotation.key_id_map import KeyIdMap


class PointcloudEpisodeAnnotationAPI(EntityAnnotationAPI):
    _method_download = 'point-clouds.episodes.annotations.info'
    _entity_ids_str = ApiField.DATASET_ID

    def download(self, dataset_id):
        """
        :param dataset_id: int
        :return: dataset episode annotation for given id, received after execution post request
        """
        response = self._api.post(self._method_download, {ApiField.DATASET_ID: dataset_id})
        return response.json()

    def download_bulk(self, dataset_id, entity_ids):
        raise NotImplementedError

    def append(self, dataset_id, ann: PointcloudEpisodeAnnotation, key_id_map: KeyIdMap = None):
        raise NotImplementedError
