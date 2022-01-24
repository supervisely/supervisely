# coding: utf-8

from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
from supervisely.api.entity_annotation.object_api import ObjectApi


class PointcloudObjectApi(ObjectApi):
    def append_bulk(self, pointcloud_id, objects: PointcloudObjectCollection, key_id_map: KeyIdMap = None):
        info = self._api.pointcloud.get_info_by_id(pointcloud_id)
        return self._append_bulk(self._api.pointcloud.tag, pointcloud_id, info.project_id, info.dataset_id, objects, key_id_map, is_pointcloud=True)

    def append_to_dataset(self, dataset_id, objects: PointcloudObjectCollection, key_id_map: KeyIdMap = None):
        project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
        return self._append_bulk(self._api.pointcloud.tag, dataset_id, project_id, dataset_id, objects, key_id_map, is_pointcloud=True)
