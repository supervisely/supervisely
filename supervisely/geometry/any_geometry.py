# coding: utf-8

from supervisely.geometry.geometry import Geometry
from supervisely.geometry.constants import ANY_SHAPE


class AnyGeometry(Geometry):
    """Placeholder geometry that accepts any shape type. Used for labels with flexible geometry. Immutable."""
    @staticmethod
    def geometry_name():
        """
        Geometry name.

        :returns: Geometry name
        :rtype: str
        """
        return ANY_SHAPE
