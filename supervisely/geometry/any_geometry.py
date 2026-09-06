# coding: utf-8

from copy import deepcopy

from supervisely.geometry.constants import ANY_SHAPE, GEOMETRY_SHAPE, GEOMETRY_TYPE
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.rectangle import Rectangle


class AnyGeometry(Geometry):
    """Placeholder geometry that accepts any shape type. Used for labels with flexible geometry. Immutable."""

    def __init__(
        self,
        data=None,
        geometry_type=None,
        sly_id=None,
        class_id=None,
        labeler_login=None,
        updated_at=None,
        created_at=None,
    ):
        super().__init__(
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
        self._data = deepcopy(data) if data is not None else {}
        self._geometry_type = geometry_type or ANY_SHAPE

    @staticmethod
    def geometry_name():
        """
        Geometry name.

        :returns: Geometry name
        :rtype: str
        """
        return ANY_SHAPE

    @property
    def raw_geometry_type(self):
        return self._geometry_type

    @classmethod
    def from_json(cls, data):
        geometry_type = data.get(GEOMETRY_TYPE, data.get(GEOMETRY_SHAPE, ANY_SHAPE))
        return cls(
            data=data,
            geometry_type=geometry_type,
            sly_id=data.get("id"),
            class_id=data.get("classId"),
            labeler_login=data.get("labelerLogin"),
            updated_at=data.get("updatedAt"),
            created_at=data.get("createdAt"),
        )

    def to_json(self):
        data = deepcopy(self._data)
        data[GEOMETRY_TYPE] = self._geometry_type
        data[GEOMETRY_SHAPE] = self._geometry_type
        self._add_creation_info(data)
        return data

    def crop(self, rect):
        return [self]

    def resize(self, in_size, out_size):
        return self

    def scale(self, factor):
        return self

    def translate(self, drow, dcol):
        return self

    def rotate(self, rotator):
        return self

    def fliplr(self, img_size):
        return self

    def flipud(self, img_size):
        return self

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        return None

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        return None

    @property
    def area(self):
        return 0

    def to_bbox(self):
        return Rectangle(0, 0, 0, 0)
