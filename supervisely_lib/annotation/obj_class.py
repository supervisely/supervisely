# coding: utf-8

from copy import deepcopy
from typing import List
from supervisely_lib.imaging.color import random_rgb, rgb2hex, hex2rgb, _validate_color
from supervisely_lib.io.json import JsonSerializable
from supervisely_lib.collection.key_indexed_collection import KeyObject
from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.any_geometry import AnyGeometry
from supervisely_lib._utils import take_with_default
from supervisely_lib.annotation.json_geometries_map import GET_GEOMETRY_FROM_STR


class ObjClassJsonFields:
    ID = 'id'
    NAME = 'title'
    GEOMETRY_TYPE = 'shape'
    COLOR = 'color'
    GEOMETRY_CONFIG = 'geometry_config'
    HOTKEY = 'hotkey'


class ObjClass(KeyObject, JsonSerializable):
    def __init__(self, name: str, geometry_type: type, color: List[int]=None, geometry_config: dict=None, sly_id=None,
                 hotkey: str = None):
        """
        Class of objects (person, car, etc) with necessary properties: name, type of geometry (Polygon, Rectangle, ...)
        and RGB color. Only one class can be associated with Label.


        Args:
            name: string name of the class (person, car, apple, etc)
            geometry_type: type of the geometry. Geometry defines the shape for all Labels of this ObjClass:
                Polygon, Rectangle, Bitmap, Polyline, Point
            color: [R, G, B]
            geometry_config: additional settings of the geometry that is associated with ObjClass
        Returns:
            ObjClass instance
        """
        self._name = name
        self._geometry_type = geometry_type
        self._color = random_rgb() if color is None else deepcopy(color)
        self._geometry_config = deepcopy(take_with_default(geometry_config, {}))
        self._sly_id = sly_id
        self._hotkey = take_with_default(hotkey, "")
        _validate_color(self._color)

    @property
    def name(self):
        """
        Returns:
            string name of the ObjectClass
        """
        return self._name

    def key(self):
        """
        Used as a key in ObjClassCollection (like key in dict)
        Returns:
            string name of the ObjectClass
        """
        return self.name

    @property
    def geometry_type(self):
        """
        Returns:
            type of the geometry that is associated with ObjClass (Polygon, Rectangle, Bitmap, etc)
        """
        return self._geometry_type

    @property
    def geometry_config(self):
        """
        Returns:
            additional settings of the geometry that is associated with ObjClass
        """
        return deepcopy(self._geometry_config)

    @property
    def color(self):
        """
        Returns:
            [R, G, B]
        """
        return deepcopy(self._color)

    @property
    def sly_id(self):
        return self._sly_id

    @property
    def hotkey(self):
        return self._hotkey

    def to_json(self) -> dict:
        """
        Converts object to json serializable dictionary. See Supervisely Json format explanation here:
        https://docs.supervise.ly/ann_format/

        Returns:
            json serializable dictionary
        """
        res = {
            ObjClassJsonFields.NAME: self.name,
            ObjClassJsonFields.GEOMETRY_TYPE: self.geometry_type.geometry_name(),
            ObjClassJsonFields.COLOR: rgb2hex(self.color),
            ObjClassJsonFields.GEOMETRY_CONFIG: self.geometry_type.config_to_json(self._geometry_config)
        }
        if self.sly_id is not None:
            res[ObjClassJsonFields.ID] = self.sly_id
        if self._hotkey is not None:
            res[ObjClassJsonFields.HOTKEY] = self.hotkey
        return res

    @classmethod
    def from_json(cls, data: dict) -> 'ObjClass':
        """
        Creates object from json serializable dictionary. See Supervisely Json format explanation here:
        https://docs.supervise.ly/ann_format/

        Returns:
            ObjClass
        """
        name = data[ObjClassJsonFields.NAME]
        geometry_type = GET_GEOMETRY_FROM_STR(data[ObjClassJsonFields.GEOMETRY_TYPE])
        color = hex2rgb(data[ObjClassJsonFields.COLOR])
        geometry_config = geometry_type.config_from_json(data.get(ObjClassJsonFields.GEOMETRY_CONFIG))
        sly_id = data.get(ObjClassJsonFields.ID, None)
        hotkey = data.get(ObjClassJsonFields.HOTKEY, "")
        return cls(name=name, geometry_type=geometry_type, color=color, geometry_config=geometry_config, sly_id=sly_id,
                   hotkey=hotkey)

    def __eq__(self, other: 'ObjClass'):
        return isinstance(other, ObjClass) and \
               self.name == other.name and \
               (self.geometry_type == other.geometry_type or AnyGeometry in [self.geometry_type, other.geometry_type]) and \
               self.geometry_config == other.geometry_config

    def __ne__(self, other: 'ObjClass'):
        return not self == other

    def __str__(self):  # Is need show geometry settings here?
        return '{:<7s}{:<10}{:<7s}{:<13}{:<7s}{:<15}{:<16s}{:<16}{:<7s}{:<7}'.format(
            'Name:', self.name,
            'Shape:', self.geometry_type.__name__,
            'Color:', str(self.color),
            'Geom. settings:', str(self.geometry_config),
            'Hotkey', self.hotkey
        )

    @classmethod
    def get_header_ptable(cls):
        return ['Name', 'Shape', 'Color', 'Hotkey']  # Is need show geometry settings here?

    def get_row_ptable(self):
        return [self.name, self.geometry_type.__name__, self.color, self.hotkey]

    def clone(self, name: str = None, geometry_type: Geometry = None, color: List[int] = None,
              geometry_config: dict = None, sly_id=None, hotkey=None) -> 'ObjClass':
        """
        Creates object duplicate. Defined arguments replace corresponding original values.

        Args:
            see __init__ method
        Returns:
            new instance of class
        """
        return ObjClass(name=take_with_default(name, self.name),
                        geometry_type=take_with_default(geometry_type, self.geometry_type),
                        color=take_with_default(color, self.color),
                        geometry_config=take_with_default(geometry_config, self.geometry_config),
                        sly_id=take_with_default(sly_id, self.sly_id),
                        hotkey=take_with_default(hotkey, self.hotkey))
