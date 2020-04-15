# coding: utf-8

from copy import deepcopy

from supervisely_lib.geometry.constants import X, Y, Z, POSITION, ROTATION, DIMENTIONS
from supervisely_lib.geometry.geometry import Geometry


class Vector3d:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def to_json(self):
        return {X: self.x, Y: self.y, Z: self.z}

    @classmethod
    def from_json(cls, data):
        x = data[X]
        y = data[Y]
        z = data[Z]
        return cls(x, y, z)

    def clone(self):
        return deepcopy(self)


class Cuboid3d(Geometry):
    @staticmethod
    def geometry_name():
        return 'cuboid_3d'

    def __init__(self, position: Vector3d, rotation: Vector3d, dimensions: Vector3d):
        if type(position) is not Vector3d:
            raise TypeError("\"position\" param has to be of type {!r}".format(type(Vector3d)))
        if type(rotation) is not Vector3d:
            raise TypeError("\"rotation\" param has to be of type {!r}".format(type(Vector3d)))
        if type(dimensions) is not Vector3d:
            raise TypeError("\"dimensions\" param has to be of type {!r}".format(type(Vector3d)))

        self._position = position
        self._rotation = rotation
        self._dimensions = dimensions

    @property
    def position(self):
        return self._position.clone()

    @property
    def rotation(self):
        return self._rotation.clone()

    @property
    def dimensions(self):
        return self._dimensions.clone()

    def to_json(self):
        return {POSITION: self.position.to_json(),
                ROTATION: self.rotation.to_json(),
                DIMENTIONS: self.dimensions.to_json()}

    @classmethod
    def from_json(cls, data):
        position = Vector3d.from_json(data[POSITION])
        rotation = Vector3d.from_json(data[ROTATION])
        dimentions = Vector3d.from_json(data[DIMENTIONS])
        return cls(position, rotation, dimentions)
