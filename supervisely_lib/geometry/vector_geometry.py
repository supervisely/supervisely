# coding: utf-8

from copy import deepcopy
import cv2
import numpy as np

from supervisely_lib.geometry.constants import EXTERIOR, INTERIOR, POINTS
from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.point import Point, points_to_row_col_list
from supervisely_lib.geometry.rectangle import Rectangle


class VectorGeometry(Geometry):
    def __init__(self, exterior, interior):
        if not (isinstance(exterior, list) and all(isinstance(p, Point) for p in exterior)):
            raise TypeError('Argument "exterior" must be list of "Point" objects!')

        if not isinstance(interior, list) or \
            not all(isinstance(c, list) for c in interior) or \
                not all(isinstance(p, Point) for c in interior for p in c):
            raise TypeError('Argument "interior" must be list of list of "Point" objects!')

        self._exterior = deepcopy(exterior)
        self._interior = deepcopy(interior)

    def to_json(self):
        packed_obj = {
            POINTS: {
                EXTERIOR: points_to_row_col_list(self._exterior, flip_row_col_order=True),
                INTERIOR: [points_to_row_col_list(i, flip_row_col_order=True) for i in self._interior]
            }
        }
        return packed_obj

    @property
    def exterior(self):
        return deepcopy(self._exterior)

    @property
    def exterior_np(self):
        return np.array(points_to_row_col_list(self._exterior), dtype=np.int64)

    @property
    def interior(self):
        return deepcopy(self._interior)

    @property
    def interior_np(self):
        return [np.array(points_to_row_col_list(i), dtype=np.int64) for i in self._interior]

    def _transform(self, transform_fn):
        result = deepcopy(self)
        result._exterior = [transform_fn(p) for p in self._exterior]
        result._interior = [[transform_fn(p) for p in i] for i in self._interior]
        return result

    def resize(self, in_size, out_size):
        return self._transform(lambda p: p.resize(in_size, out_size))

    def scale(self, factor):
        return self._transform(lambda p: p.scale(factor))

    def translate(self, drow, dcol):
        return self._transform(lambda p: p.translate(drow, dcol))

    def rotate(self, rotator):
        return self._transform(lambda p: p.rotate(rotator))

    def fliplr(self, img_size):
        return self._transform(lambda p: p.fliplr(img_size))

    def flipud(self, img_size):
        return self._transform(lambda p: p.flipud(img_size))

    def to_bbox(self):
        exterior_np = self.exterior_np
        rows, cols = exterior_np[:, 0], exterior_np[:, 1]
        return Rectangle(top=round(min(rows).item()), left=round(min(cols).item()), bottom=round(max(rows).item()),
                         right=round(max(cols).item()))

    def draw(self, bitmap: np.ndarray, color, thickness=1):
        self.draw_contour(bitmap, color, thickness=thickness)  # due to cv2

    def draw_contour(self, bitmap: np.ndarray, color, thickness=1):
        # TODO move most of implementation here
        raise NotImplementedError('aaaaa')

    @staticmethod
    def _approx_ring_dp(ring, epsilon, closed):
        new_ring = cv2.approxPolyDP(ring.astype(np.int32), epsilon, closed)
        new_ring = np.squeeze(new_ring, 1)
        if len(new_ring) < 3 and closed:
            new_ring = ring.astype(np.int32)
        return new_ring

    def approx_dp(self, epsilon):
        raise NotImplementedError()
