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
from supervisely_lib.geometry.constants import LABELER_LOGIN, UPDATED_AT, CREATED_AT, CLASS_ID


class VideoObject(KeyObject):
    '''
    This is a class for creating and using VideoObject objects for videos
    '''
    def __init__(self, obj_class: ObjClass, tags: VideoTagCollection = None, key=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        '''
        :param obj_class: ObjClass class object
        :param tags: VideoTagCollection
        :param key: uuid class object
        '''
        self.labeler_login = labeler_login
        self.updated_at = updated_at
        self.created_at = created_at

        self._obj_class = obj_class
        self._key = take_with_default(key, uuid.uuid4())
        self._tags = take_with_default(tags, VideoTagCollection())

    def _add_creation_info(self, d):
        if self.labeler_login is not None:
            d[LABELER_LOGIN] = self.labeler_login
        if self.updated_at is not None:
            d[UPDATED_AT] = self.updated_at
        if self.created_at is not None:
            d[CREATED_AT] = self.created_at

    @property
    def obj_class(self):
        return self._obj_class

    def key(self):
        return self._key

    @property
    def tags(self):
        return self._tags.clone()

    def add_tag(self, tag: VideoTag):
        '''
        Add given tag to VideoTagCollection
        :param tag: VideoTag class object
        :return: VideoObject class object
        '''
        return self.clone(tags=self._tags.add(tag))

    def add_tags(self, tags: list):
        '''
        Add given tags to VideoTagCollection
        :param tags: list of VideoTag class objects
        :return: VideoObject class object
        '''
        return self.clone(tags=self._tags.add_items(tags))

    def to_json(self, key_id_map: KeyIdMap = None):
        '''
        The function to_json convert VideoObject class object to json format
        :param key_id_map: KeyIdMap class object
        :return: VideoObject in json format
        '''
        data_json = {
            KEY: self.key().hex,
            LabelJsonFields.OBJ_CLASS_NAME: self.obj_class.name,
            LabelJsonFields.TAGS: self.tags.to_json(key_id_map)
        }

        if key_id_map is not None:
            item_id = key_id_map.get_object_id(self.key())
            if item_id is not None:
                data_json[ID] = item_id

        self._add_creation_info(data_json)
        return data_json

    @classmethod
    def from_json(cls, data, project_meta: ProjectMeta, key_id_map: KeyIdMap = None):
        '''
        The function from_json convert VideoObject from json format to VideoObject class object. Raise error if object class name is not found in the given project meta
        :param data: input VideoObject in json format
        :param project_meta: ProjectMeta class object
        :param key_id_map: KeyIdMap class object
        :return: VideoObject class object
        '''
        obj_class_name = data[LabelJsonFields.OBJ_CLASS_NAME]
        obj_class = project_meta.get_obj_class(obj_class_name)
        if obj_class is None:
            raise RuntimeError(f'Failed to deserialize a object from JSON: class name {obj_class_name!r} '
                               f'was not found in the given project meta.')

        key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_object(key, data.get(ID, None))

        class_id = data.get(CLASS_ID, None)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)

        return cls(obj_class=obj_class,
                   key=key,
                   tags=VideoTagCollection.from_json(data[LabelJsonFields.TAGS], project_meta.tag_metas),
                   class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

    def clone(self, obj_class: ObjClass=None, tags: VideoTagCollection = None, key=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        '''
        :param obj_class: ObjClass
        :param tags: VideoTagCollection
        :param key: uuid class object
        :return: VideoObject class object
        '''
        return self.__class__(obj_class=take_with_default(obj_class, self.obj_class),
                              key=take_with_default(key, self._key),
                              tags=take_with_default(tags, self.tags),
                              class_id=take_with_default(class_id, self.class_id),
                              labeler_login=take_with_default(labeler_login, self.labeler_login),
                              updated_at=take_with_default(updated_at, self.updated_at),
                              created_at=take_with_default(created_at, self.created_at))

