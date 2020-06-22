# coding: utf-8

from supervisely_lib.api.module_api import ApiField
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.pointcloud_annotation.pointcloud_annotation import PointcloudAnnotation
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

from supervisely_lib.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI


class PointcloudAnnotationAPI(EntityAnnotationAPI):
    _method_download_bulk = 'point-clouds.annotations.bulk.info'
    _entity_ids_str = ApiField.POINTCLOUD_IDS

    def download(self, pointcloud_id):
        '''
        :param pointcloud_id: int
        :return: list of content of pointcloud object with given id, received after execution post request
        '''
        info = self._api.pointcloud.get_info_by_id(pointcloud_id)
        return self._download(info.dataset_id, pointcloud_id)

    def append(self, pointcloud_id, ann: PointcloudAnnotation, key_id_map: KeyIdMap = None):
        info = self._api.pointcloud.get_info_by_id(pointcloud_id)

        new_objects = []
        for object_3d in ann.objects:
            if key_id_map is not None and key_id_map.get_object_id(object_3d.key()) is not None:
                # object already uploaded
                continue
            new_objects.append(object_3d)

        self._append(self._api.pointcloud.tag, self._api.pointcloud.object, self._api.pointcloud.figure,
                     info.project_id, info.dataset_id, pointcloud_id,
                     ann.tags, PointcloudObjectCollection(new_objects), ann.figures, key_id_map)
