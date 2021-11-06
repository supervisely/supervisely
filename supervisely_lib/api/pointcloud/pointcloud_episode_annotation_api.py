# coding: utf-8

from supervisely_lib.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely_lib.api.pointcloud.pointcloud_annotation_api import PointcloudAnnotationAPI
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.pointcloud_annotation.pointcloud_episode_annotation import PointcloudEpisodeAnnotation
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.pointcloud_annotation.pointcloud_annotation import PointcloudAnnotation
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection


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


    def append(self, project_id, dataset_id, pointcloud_id, ann: PointcloudAnnotation, key_id_map: KeyIdMap = None):
        new_objects = []
        for object_3d in ann.objects:
            if key_id_map is not None and key_id_map.get_object_id(object_3d.key()) is not None:
                # object already uploaded
                continue
            new_objects.append(object_3d)

        self._append(self._api.pointcloud.tag, self._api.pointcloud.object, self._api.pointcloud.figure,
                     project_id, pointcloud_id,
                     ann.tags, PointcloudObjectCollection(new_objects), ann.figures, key_id_map)
