# coding: utf-8

from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.constants import ANY_SHAPE


class AnyGeometry(Geometry):
    """
    AnyGeometry for a single :class:`Label<supervisely_lib.annotation.label.Label>`. :class:`AnyGeometry<AnyGeometry>` class object is immutable.
    """
    @staticmethod
    def geometry_name():
        """
        Geometry name.

        :return: Geometry name
        :rtype: :class:`str`
        """
        return ANY_SHAPE
