# coding: utf-8

import uuid

from supervisely_lib.video_annotation.video_object import VideoObject
from supervisely_lib.annotation.label import LabelJsonFields
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.video_annotation.constants import KEY, ID
from supervisely_lib.video_annotation.video_tag_collection import VideoTagCollection
from supervisely_lib.video_annotation.key_id_map import KeyIdMap


class PointcloudObject(VideoObject):
    @classmethod
    def from_json(cls, data, project_meta: ProjectMeta, key_id_map: KeyIdMap = None):
        obj_class_name = data[LabelJsonFields.OBJ_CLASS_NAME]
        obj_class = project_meta.get_obj_class(obj_class_name)
        if obj_class is None:
            raise RuntimeError(f'Failed to deserialize a object from JSON: class name {obj_class_name!r} '
                               f'was not found in the given project meta.')

        object_id = data.get(ID, None)
        existing_key = key_id_map.get_object_key(object_id) if object_id is not None else None
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

        return cls(obj_class=obj_class,
                   key=key,
                   tags=VideoTagCollection.from_json(data[LabelJsonFields.TAGS], project_meta.tag_metas))