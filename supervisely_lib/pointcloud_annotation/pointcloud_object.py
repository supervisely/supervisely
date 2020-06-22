# coding: utf-8

import uuid

from supervisely_lib.video_annotation.video_object import VideoObject
from supervisely_lib.annotation.label import LabelJsonFields
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.video_annotation.constants import KEY, ID
from supervisely_lib.video_annotation.video_tag_collection import VideoTagCollection
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.geometry.constants import LABELER_LOGIN, UPDATED_AT, CREATED_AT, CLASS_ID


class PointcloudObject(VideoObject):
    '''
    This is a class for creating and using PointcloudObject objects for videos
    '''
    @classmethod
    def from_json(cls, data, project_meta: ProjectMeta, key_id_map: KeyIdMap = None):
        '''
        The function from_json convert PointcloudObject from json format to PointcloudObject class object. Raise error if object class name is not found in the given project meta
        :param data: input PointcloudObject in json format
        :param project_meta: ProjectMeta class object
        :param key_id_map: KeyIdMap class object
        :return: PointcloudObject class object
        '''
        obj_class_name = data[LabelJsonFields.OBJ_CLASS_NAME]
        obj_class = project_meta.get_obj_class(obj_class_name)
        if obj_class is None:
            raise RuntimeError(f'Failed to deserialize a object from JSON: class name {obj_class_name!r} '
                               f'was not found in the given project meta.')

        object_id = data.get(ID, None)

        existing_key = None
        if object_id is not None and key_id_map is not None:
            existing_key = key_id_map.get_object_key(object_id)
        json_key = uuid.UUID(data[KEY]) if KEY in data else None
        if (existing_key is not None) and (json_key is not None) and (existing_key != json_key):
            raise RuntimeError("Object id = {!r}: existing_key {!r} != json_key {!r}"
                               .format(object_id, existing_key, json_key))

        if existing_key is not None:
            key = existing_key
        elif json_key is not None:
            key = json_key
        else:
            key = uuid.uuid4()

        if key_id_map is not None and existing_key is None:
            key_id_map.add_object(key, object_id)

        class_id = data.get(CLASS_ID, None)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)

        return cls(obj_class=obj_class,
                   key=key,
                   tags=VideoTagCollection.from_json(data[LabelJsonFields.TAGS], project_meta.tag_metas),
                   class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)