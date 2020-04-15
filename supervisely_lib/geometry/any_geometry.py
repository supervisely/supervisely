# coding: utf-8

from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.constants import ANY_SHAPE

class AnyGeometry(Geometry):
    @staticmethod
    def geometry_name():
        return ANY_SHAPE

