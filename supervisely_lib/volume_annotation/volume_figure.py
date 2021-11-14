# coding: utf-8

import uuid

from supervisely_lib._utils import take_with_default
from supervisely_lib.geometry.any_geometry import AnyGeometry
from supervisely_lib.annotation.json_geometries_map import GET_GEOMETRY_FROM_STR
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.video_annotation.key_id_map import KeyIdMap

from supervisely_lib.geometry.constants import LABELER_LOGIN, UPDATED_AT, CREATED_AT, CLASS_ID
from supervisely_lib.volume_annotation.volume_object_collection import VolumeObjectCollection
from supervisely_lib.volume_annotation import constants as const


class VolumeFigure:
    def __init__(self, volume_object, geometry, meta, key=None, priority=None, bitmap=None, description=None,
                 class_id=None, labeler_login=None, updated_at=None, created_at=None):
        self._volume_object = volume_object
        self._priority = priority
        self._meta = meta
        self._set_geometry_inplace(geometry)
        self._bitmap = bitmap
        self._description = description
        self._key = take_with_default(key, uuid.uuid4())
        self.class_id = class_id
        self.labeler_login = labeler_login
        self.updated_at = updated_at
        self.created_at = created_at

    def _set_geometry_inplace(self, geometry):
        self._geometry = geometry
        self._validate_geometry_type()
        self._validate_geometry()

    @property
    def volume_object(self):
        return self._volume_object

    @property
    def parent_object(self):
        return self._volume_object

    @property
    def geometry(self):
        return self._geometry

    @property
    def priority(self):
        return self._priority

    @property
    def meta(self):
        return self._meta

    @property
    def slice_index(self):
        if self.meta:
            return self.meta.get(const.SLICE_INDEX, None)
        else:
            return None

    @property
    def normal(self):
        if self.meta:
            return self.meta.get(const.NORMAL, None)
        else:
            return None

    def key(self):
        return self._key

    def _validate_geometry(self):
        self._geometry.validate(self.parent_object.obj_class.geometry_type.geometry_name(),
                                self.parent_object.obj_class.geometry_config)

    def _validate_geometry_type(self):
        if self.parent_object.obj_class.geometry_type != AnyGeometry:
            if type(self._geometry) is not self.parent_object.obj_class.geometry_type:
                raise RuntimeError("Input geometry type {!r} != geometry type of ObjClass {}"
                                   .format(type(self._geometry), self.parent_object.obj_class.geometry_type))

    @classmethod
    def from_json(cls, data, objects: VolumeObjectCollection, key_id_map: KeyIdMap = None):
        object_id = data.get(ApiField.OBJECT_ID, None)
        object_key = None
        if const.OBJECT_KEY in data:
            object_key = uuid.UUID(data[const.OBJECT_KEY])

        if object_id is None and object_key is None:
            raise RuntimeError("Figure can not be deserialized from json: object_id or object_key are not found")

        if object_key is None:
            if key_id_map is None:
                raise RuntimeError("Figure can not be deserialized: key_id_map is None")
            object_key = key_id_map.get_object_key(object_id)
            if object_key is None:
                raise RuntimeError("Object with id={!r} not found in key_id_map".format(object_id))

        object = objects.get(object_key)
        if object is None:
            raise RuntimeError(
                "Figure can not be deserialized: corresponding object {!r} not found in ObjectsCollection".format(
                    object_key.hex))

        shape_str = data[ApiField.GEOMETRY_TYPE]
        geometry_json = data[const.POINTS]

        shape = GET_GEOMETRY_FROM_STR(shape_str)
        geometry = shape.from_json(geometry_json)

        key = uuid.UUID(data[const.KEY]) if const.KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_figure(key, data.get(const.ID, None))

        meta = data.get(const.META, None)
        description = data.get(const.DESCRIPTION, '')
        priority = data.get(const.PRIORITY, None)
        bitmap = data.get(const.BITMAP, None)

        class_id = data.get(CLASS_ID, None)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)

        return cls(object,
                   geometry, meta,
                   key=key,
                   priority=priority,
                   description=description, bitmap=bitmap,
                   class_id=class_id, labeler_login=labeler_login,
                   updated_at=updated_at, created_at=created_at)

    def to_json(self, key_id_map=None, save_meta=False):
        raise NotImplementedError('yet')

    def clone(self, volume_object=None, geometry=None, meta=None, key=None, priority=None, bitmap=None,
              description=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        return self.__class__(volume_object=take_with_default(volume_object, self.parent_object),
                              geometry=take_with_default(geometry, self.geometry),
                              meta=take_with_default(meta, self.meta),
                              key=take_with_default(key, self._key),
                              priority=take_with_default(priority, self.priority),
                              bitmap=take_with_default(bitmap, self._bitmap),
                              description=take_with_default(description, self._description),
                              class_id=take_with_default(class_id, self.class_id),
                              labeler_login=take_with_default(labeler_login, self.labeler_login),
                              updated_at=take_with_default(updated_at, self.updated_at),
                              created_at=take_with_default(created_at, self.created_at))
