# coding: utf-8

from supervisely_lib.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.pointcloud_annotation.pointcloud_episode_annotation import PointcloudEpisodeAnnotation
from supervisely_lib.video_annotation.key_id_map import KeyIdMap


class PointcloudEpisodeAnnotationAPI(EntityAnnotationAPI):
    _method_download = 'point-clouds.episodes.annotations.info'
    _entity_ids_str = ApiField.POINTCLOUD_IDS

    def download(self, dataset_id):
        """
        :param dataset_id: int
        :return: dataset episode annotation for given id, received after execution post request
        """
        response = self._api.post(self._method_download, {ApiField.DATASET_ID: dataset_id})
        return response.json()[0]

    def download_bulk(self, dataset_id, entity_ids):
        raise RuntimeError('Not supported for episodes')

    def append(self, dataset_id, ann: PointcloudEpisodeAnnotation, frame_to_pointcloud_ids,
               key_id_map: KeyIdMap = None):
        if key_id_map is None:
            # create for internal purposes (to link figures and tags to objects)
            key_id_map = KeyIdMap()

        figures = []
        pointcloud_ids = []
        for i, frame in enumerate(ann.frames):
            for fig in frame.figures:
                figures.append(fig)
                pointcloud_ids.append(frame_to_pointcloud_ids[i])

        self._api.pointcloud_episode.object.append_to_dataset(dataset_id, ann.objects, key_id_map)
        self._api.pointcloud_episode.figure.append_to_dataset(dataset_id, figures, pointcloud_ids, key_id_map)

