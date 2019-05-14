# coding: utf-8

import cv2
import numpy as np

from supervisely_lib.geometry.constants import EXTERIOR, INTERIOR, POINTS
from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.point_location import PointLocation, points_to_row_col_list
from supervisely_lib.geometry import validation


# @TODO: validation
class Rectangle(Geometry):
    @staticmethod
    def geometry_name():
        return 'rectangle'

    def __init__(self, top, left, bottom, right):
        """
        Float-type coordinates will be deprecated soon.
        Args:
            top: minimal vertical value
            left: minimal horizontal value
            bottom: maximal vertical value
            right: maximal horizontal value
        """
        if top > bottom:
            raise ValueError('Rectangle "top" argument must have less or equal value then "bottom"!')

        if left > right:
            raise ValueError('Rectangle "left" argument must have less or equal value then "right"!')

        self._points = [PointLocation(row=top, col=left), PointLocation(row=bottom, col=right)]

    """
    Implementation of all methods from Geometry
    """

    def to_json(self):
        packed_obj = {
            POINTS: {
                EXTERIOR: points_to_row_col_list(self._points, flip_row_col_order=True),
                INTERIOR: []
            }
        }
        return packed_obj

    @classmethod
    def from_json(cls, data):
        validation.validate_geometry_points_fields(data)
        exterior = data[POINTS][EXTERIOR]
        if len(exterior) != 2:
            raise ValueError('"exterior" field must contain exactly two points to create Rectangle object.')
        [top, bottom] = sorted([exterior[0][1], exterior[1][1]])
        [left, right] = sorted([exterior[0][0], exterior[1][0]])
        return cls(top=top, left=left, bottom=bottom, right=right)

    def crop(self, other):
        top = max(self.top, other.top)
        left = max(self.left, other.left)
        bottom = min(self.bottom, other.bottom)
        right = min(self.right, other.right)
        is_valid = (bottom >= top) and (left <= right)
        return [Rectangle(top=top, left=left, bottom=bottom, right=right)] if is_valid else []

    def _transform(self, transform_fn):
        transformed_corners = [transform_fn(p) for p in self.corners]
        rows, cols = zip(*points_to_row_col_list(transformed_corners))
        return Rectangle(top=round(min(rows)), left=round(min(cols)), bottom=round(max(rows)), right=round(max(cols)))

    @property
    def corners(self):
        return [PointLocation(row=self.top, col=self.left), PointLocation(row=self.top, col=self.right),
                PointLocation(row=self.bottom, col=self.right), PointLocation(row=self.bottom, col=self.left)]

    def rotate(self, rotator):
        return self._transform(lambda p: rotator.transform_point(p))

    def resize(self, in_size, out_size):
        return self._transform(lambda p: p.resize(in_size, out_size))

    def scale(self, factor):
        return self._transform(lambda p: p.scale(factor))

    def translate(self, drow, dcol):
        return self._transform(lambda p: p.translate(drow, dcol))

    def fliplr(self, img_size):
        img_width = img_size[1]
        return Rectangle(top=self.top, left=(img_width - self.right), bottom=self.bottom, right=(img_width - self.left))

    def flipud(self, img_size):
        img_height = img_size[0]
        return Rectangle(top=(img_height - self.bottom), left=self.left, bottom=(img_height - self.top),
                         right=self.right)

    def _draw_impl(self, bitmap: np.ndarray, color, thickness=1, config=None):
        self._draw_contour_impl(bitmap, color, thickness=cv2.FILLED, config=config)  # due to cv2

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        cv2.rectangle(bitmap, pt1=(self.left, self.top), pt2=(self.right, self.bottom), color=color,
                      thickness=thickness)

    def to_bbox(self):
        return self.clone()

    @property
    def area(self):
        return float(self.width * self.height)

    @classmethod
    def from_array(cls, arr):
        return cls(top=0, left=0, bottom=arr.shape[0] - 1, right=arr.shape[1] - 1)

    # TODO re-evaluate whether we need this, looks trivial.
    @classmethod
    def from_size(cls, size: tuple):
        return cls(0, 0, size[0] - 1, size[1] - 1)

    @classmethod
    def from_geometries_list(cls, geometries):
        bboxes = [g.to_bbox() for g in geometries]
        top = min(bbox.top for bbox in bboxes)
        left = min(bbox.left for bbox in bboxes)
        bottom = max(bbox.bottom for bbox in bboxes)
        right = max(bbox.right for bbox in bboxes)
        return cls(top=top, left=left, bottom=bottom, right=right)

    @property
    def left(self):
        return self._points[0].col

    @property
    def right(self):
        return self._points[1].col

    @property
    def top(self):
        return self._points[0].row

    @property
    def bottom(self):
        return self._points[1].row

    @property
    def center(self):
        return PointLocation(row=(self.top + self.bottom) // 2, col=(self.left + self.right) // 2)

    @property
    def width(self):
        return self.right - self.left + 1

    @property
    def height(self):
        return self.bottom - self.top + 1

    def contains(self, rect):
        return (self.left <= rect.left and
                self.right >= rect.right and
                self.top <= rect.top and
                self.bottom >= rect.bottom)

    def contains_point_location(self, pt: PointLocation):
        return (self.left <= pt.col <= self.right) and (self.top <= pt.row <= self.bottom)

    def to_size(self):
        return self.height, self.width

    def get_cropped_numpy_slice(self, data: np.ndarray) -> np.ndarray:
        return data[self.top:(self.bottom+1), self.left:(self.right+1), ...]
