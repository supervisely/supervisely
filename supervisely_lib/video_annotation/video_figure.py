# coding: utf-8

import uuid
from bidict import bidict

from supervisely_lib.api.module_api import ApiField

from supervisely_lib._utils import take_with_default
from supervisely_lib.geometry.any_geometry import AnyGeometry
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.video_annotation.video_object_collection import VideoObjectCollection
from supervisely_lib.video_annotation.constants import ID, KEY, OBJECT_ID, OBJECT_KEY, META
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.annotation.json_geometries_map import GET_GEOMETRY_FROM_STR
from supervisely_lib.video_annotation.key_id_map import KeyIdMap


class OutOfImageBoundsExtension(Exception):
    pass


class VideoFigure:
    def __init__(self, video_object, geometry, frame_index, key=None):
        self._video_object = video_object
        self._set_geometry_inplace(geometry)
        self._frame_index = frame_index
        self._key = take_with_default(key, uuid.uuid4())

    def _set_geometry_inplace(self, geometry):
        self._geometry = geometry
        self._validate_geometry_type()
        self._validate_geometry()

    @property
    def video_object(self):
        return self._video_object

    @property
    def parent_object(self):
        return self._video_object

    @property
    def geometry(self):
        return self._geometry

    @property
    def frame_index(self):
        return self._frame_index

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

    def to_json(self, key_id_map=None, save_meta=False):
        data_json = {
            KEY: self.key().hex,
            OBJECT_KEY: self.parent_object.key().hex,
            ApiField.GEOMETRY_TYPE: self.geometry.geometry_name(),
            ApiField.GEOMETRY: self.geometry.to_json()
        }

        if key_id_map is not None:
            item_id = key_id_map.get_figure_id(self.key())
            if item_id is not None:
                data_json[ID] = item_id

            object_id = key_id_map.get_object_id(self.parent_object.key())
            if object_id is not None:
                data_json[OBJECT_ID] = object_id
        if save_meta is True:
            data_json[ApiField.META] = {ApiField.FRAME: self.frame_index}
        return data_json

    @classmethod
    def from_json(cls, data, objects: VideoObjectCollection, frame_index, key_id_map: KeyIdMap = None):
        object_id = data.get(OBJECT_ID, None)
        object_key = None
        if OBJECT_KEY in data:
            object_key = uuid.UUID(data[OBJECT_KEY])

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
            raise RuntimeError("Figure can not be deserialized: corresponding object {!r} not found in ObjectsCollection".format(object_key.hex))

        shape_str = data[ApiField.GEOMETRY_TYPE]
        geometry_json = data[ApiField.GEOMETRY]

        shape = GET_GEOMETRY_FROM_STR(shape_str)
        geometry = shape.from_json(geometry_json)

        key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_figure(key, data.get(ID, None))
        return cls(object, geometry, frame_index, key)

    def clone(self, video_object=None, geometry=None, frame_index=None, key=None):
        return self.__class__(video_object=take_with_default(video_object, self.parent_object),
                              geometry=take_with_default(geometry, self.geometry),
                              frame_index=take_with_default(frame_index, self.frame_index),
                              key=take_with_default(key, self._key))

    def validate_bounds(self, img_size, _auto_correct=False):
        canvas_rect = Rectangle.from_size(img_size)
        if canvas_rect.contains(self.geometry.to_bbox()) is False:
            raise OutOfImageBoundsExtension("Figure is out of image bounds")

        if _auto_correct is True:
            geometries_after_crop = [cropped_geometry for cropped_geometry in self.geometry.crop(canvas_rect)]
            if len(geometries_after_crop) != 1:
                raise OutOfImageBoundsExtension("Several geometries after crop")
            self._set_geometry_inplace(geometries_after_crop[0])
