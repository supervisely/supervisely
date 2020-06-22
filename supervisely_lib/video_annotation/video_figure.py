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

from supervisely_lib.geometry.constants import LABELER_LOGIN, UPDATED_AT, CREATED_AT, CLASS_ID


class OutOfImageBoundsExtension(Exception):
    pass


class VideoFigure:
    def __init__(self, video_object, geometry, frame_index, key=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        self._video_object = video_object
        self._set_geometry_inplace(geometry)
        self._frame_index = frame_index
        self._key = take_with_default(key, uuid.uuid4())
        self.class_id = class_id
        self.labeler_login = labeler_login
        self.updated_at = updated_at
        self.created_at = created_at

    def _add_creation_info(self, d):
        if self.labeler_login is not None:
            d[LABELER_LOGIN] = self.labeler_login
        if self.updated_at is not None:
            d[UPDATED_AT] = self.updated_at
        if self.created_at is not None:
            d[CREATED_AT] = self.created_at

    def _set_geometry_inplace(self, geometry):
        '''
        Checks the given geometry for correctness. Raise error if given geometry type != geometry type of VideoObject class
        :param geometry: Geometry class object (Point, Rectangle etc)
        '''
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
        '''
        Checks geometry of VideoFigure class object for correctness
        '''
        self._geometry.validate(self.parent_object.obj_class.geometry_type.geometry_name(),
                                self.parent_object.obj_class.geometry_config)

    def _validate_geometry_type(self):
        '''
        Raise error if given geometry type != geometry type of VideoObject class
        '''
        if self.parent_object.obj_class.geometry_type != AnyGeometry:
            if type(self._geometry) is not self.parent_object.obj_class.geometry_type:
                raise RuntimeError("Input geometry type {!r} != geometry type of ObjClass {}"
                                   .format(type(self._geometry), self.parent_object.obj_class.geometry_type))

    def to_json(self, key_id_map=None, save_meta=False):
        '''
        The function to_json convert VideoFigure to json format
        :param key_id_map: KeyIdMap class object
        :param save_meta: bool
        :return: VideoFigure in json format
        '''
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

        self._add_creation_info(data_json)
        return data_json

    @classmethod
    def from_json(cls, data, objects: VideoObjectCollection, frame_index, key_id_map: KeyIdMap = None):
        '''
        The function from_json convert VideoFigure from json format to VideoFigure class object.
        :param data: input VideoFigure in json format
        :param objects: VideoObjectCollection
        :param frame_index: int
        :param key_id_map: KeyIdMap class object
        :return: VideoFigure class object
        '''
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

        class_id = data.get(CLASS_ID, None)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)

        return cls(object, geometry, frame_index, key,
                   class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

    def clone(self, video_object=None, geometry=None, frame_index=None, key=None,
              class_id=None, labeler_login=None, updated_at=None, created_at=None):
        return self.__class__(video_object=take_with_default(video_object, self.parent_object),
                              geometry=take_with_default(geometry, self.geometry),
                              frame_index=take_with_default(frame_index, self.frame_index),
                              key=take_with_default(key, self._key),
                              class_id=take_with_default(class_id, self.class_id),
                              labeler_login=take_with_default(labeler_login, self.labeler_login),
                              updated_at=take_with_default(updated_at, self.updated_at),
                              created_at=take_with_default(created_at, self.created_at)
                              )

    def validate_bounds(self, img_size, _auto_correct=False):
        '''
        The function validate_bounds checks if given image contains a figure. Raise error if figure is out of image bounds
        :param img_size: tuple or list of integers
        :param _auto_correct: bool
        '''
        canvas_rect = Rectangle.from_size(img_size)
        if canvas_rect.contains(self.geometry.to_bbox()) is False:
            raise OutOfImageBoundsExtension("Figure is out of image bounds")

        if _auto_correct is True:
            geometries_after_crop = [cropped_geometry for cropped_geometry in self.geometry.crop(canvas_rect)]
            if len(geometries_after_crop) != 1:
                raise OutOfImageBoundsExtension("Several geometries after crop")
            self._set_geometry_inplace(geometries_after_crop[0])
