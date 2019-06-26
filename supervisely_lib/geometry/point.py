# coding: utf-8

import cv2

from supervisely_lib.geometry.point_location import PointLocation
from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib._utils import unwrap_if_numpy


class Point(Geometry):
    def __init__(self, row, col):
        """
        Create geopmetry point in (row, col) position. Float-type coordinates will be deprecated soon.
        """
        self._row = round(unwrap_if_numpy(row))
        self._col = round(unwrap_if_numpy(col))

    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

    @classmethod
    def from_point_location(cls, pt: PointLocation):
        return cls(row=pt.row, col=pt.col)

    @property
    def point_location(self) -> PointLocation:
        return PointLocation(row=self.row, col=self.col)

    @staticmethod
    def geometry_name():
        return 'point'

    def crop(self, rect):
        return [self.clone()] if rect.contains_point_location(self.point_location) else []

    def rotate(self, rotator):
        return self.from_point_location(self.point_location.rotate(rotator))

    def resize(self, in_size, out_size):
        return self.from_point_location(self.point_location.resize(in_size, out_size))

    def fliplr(self, img_size):
        return self.from_point_location(self.point_location.fliplr(img_size))

    def flipud(self, img_size):
        return self.from_point_location(self.point_location.flipud(img_size))

    def scale(self, factor):
        return self.from_point_location(self.point_location.scale(factor))

    def translate(self, drow, dcol):
        return self.from_point_location(self.point_location.translate(drow, dcol))

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        r = round(thickness / 2)  # @TODO: relation between thickness and point radius - ???
        cv2.circle(bitmap, (self.col, self.row), radius=r, color=color, thickness=cv2.FILLED)

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        # @TODO: mb dummy operation for Point
        r = round((thickness + 1) / 2)
        cv2.circle(bitmap, (self.col, self.row), radius=r, color=color, thickness=cv2.FILLED)

    @property
    def area(self):
        return 0.0

    def to_bbox(self):
        return Rectangle(top=self.row, left=self.col, bottom=self.row, right=self.col)

    def to_json(self):
        return self.point_location.to_json()

    @classmethod
    def from_json(cls, data):
        return cls.from_point_location(PointLocation.from_json(data))




