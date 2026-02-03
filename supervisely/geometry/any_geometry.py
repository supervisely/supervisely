# coding: utf-8

from supervisely.geometry.geometry import Geometry
from supervisely.geometry.constants import ANY_SHAPE


class AnyGeometry(Geometry):
    """
    AnyGeometry for a single :class:`~supervisely.annotation.label.Label`. :class:`~supervisely.geometry.any_geometry.AnyGeometry` class object is immutable.
    """
    @staticmethod
    def geometry_name():
        """
        Geometry name.

        :returns: Geometry name
        :rtype: str
        """
        return ANY_SHAPE
