# coding: utf-8

import uuid
from bidict import bidict

from supervisely_lib.annotation.label import LabelJsonFields
from supervisely_lib.annotation.obj_class import ObjClass
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib._utils import take_with_default
from supervisely_lib.video_annotation.constants import KEY, ID, OBJECTS_MAP
from supervisely_lib.video_annotation.video_tag_collection import VideoTagCollection
from supervisely_lib.video_annotation.video_tag import VideoTag
from supervisely_lib.collection.key_indexed_collection import KeyObject
from supervisely_lib.video_annotation.key_id_map import KeyIdMap


class VideoObject(KeyObject):
    def __init__(self, obj_class: ObjClass, tags: VideoTagCollection = None, key=None):
        self._obj_class = obj_class
        self._key = take_with_default(key, uuid.uuid4())
        self._tags = take_with_default(tags, VideoTagCollection())

    @property
    def obj_class(self):
        return self._obj_class

    def key(self):
        return self._key

    @property
    def tags(self):
        return self._tags.clone()

    def add_tag(self, tag: VideoTag):
        return self.clone(tags=self._tags.add(tag))

    def add_tags(self, tags: list):
        return self.clone(tags=self._tags.add_items(tags))

    def to_json(self, key_id_map: KeyIdMap = None):
        data_json = {
            KEY: self.key().hex,
            LabelJsonFields.OBJ_CLASS_NAME: self.obj_class.name,
            LabelJsonFields.TAGS: self.tags.to_json(key_id_map)
        }

        if key_id_map is not None:
            item_id = key_id_map.get_object_id(self.key())
            if item_id is not None:
                data_json[ID] = item_id

        return data_json

    @classmethod
    def from_json(cls, data, project_meta: ProjectMeta, key_id_map: KeyIdMap = None):
        obj_class_name = data[LabelJsonFields.OBJ_CLASS_NAME]
        obj_class = project_meta.get_obj_class(obj_class_name)
        if obj_class is None:
            raise RuntimeError(f'Failed to deserialize a object from JSON: class name {obj_class_name!r} '
                               f'was not found in the given project meta.')

        key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_object(key, data.get(ID, None))

        return cls(obj_class=obj_class,
                   key=key,
                   tags=VideoTagCollection.from_json(data[LabelJsonFields.TAGS], project_meta.tag_metas))

    def clone(self, obj_class: ObjClass=None, tags: VideoTagCollection = None, key=None):
        return self.__class__(obj_class=take_with_default(obj_class, self.obj_class),
                              key=take_with_default(key, self._key),
                              tags=take_with_default(tags, self.tags))

