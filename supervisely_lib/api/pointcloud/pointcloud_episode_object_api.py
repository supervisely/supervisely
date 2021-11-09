from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
from supervisely_lib.api.entity_annotation.object_api import ObjectApi
from supervisely_lib.api.module_api import ApiField


class PointcloudEpisodeObjectAPI(ObjectApi):
    def append_bulk(self, dataset_id, objects: PointcloudObjectCollection, key_id_map: KeyIdMap = None):
        info = self._api.dataset.get_info_by_id(dataset_id)

        objcls_name_id_map = self._api.object_class.get_name_to_id_map(info.project_id)
        items = []
        for obj in objects:
            new_obj = {ApiField.CLASS_ID: objcls_name_id_map[obj.obj_class.name]}
            items.append(new_obj)

        response = self._api.post('annotation-objects.bulk.add', {ApiField.DATASET_ID: dataset_id,
                                                                  ApiField.ANNOTATION_OBJECTS: items})
        ids = [obj[ApiField.ID] for obj in response.json()]
        KeyIdMap.add_objects_to(key_id_map, [obj.key() for obj in objects], ids)
        # add tags to objects
        self._api.pointcloud.tag.append_to_objects(dataset_id, info.project_id, objects, key_id_map)
        return ids
