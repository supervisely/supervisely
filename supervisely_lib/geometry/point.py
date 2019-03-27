# coding: utf-8
import cv2

from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.constants import EXTERIOR, INTERIOR, POINTS
from supervisely_lib.geometry import validation
from supervisely_lib.imaging import image


class Point(Geometry):
    @staticmethod
    def geometry_name():
        return 'point'

    def __init__(self, row, col):
        """
        Create point in (row, col) position. Float-type coordinates will be deprecated soon.
        """
        self._row = round(row)
        self._col = round(col)

    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

    def to_json(self):
        packed_obj = {
            POINTS: {
                EXTERIOR: [[self.col, self.row]],
                INTERIOR: []
            }
        }
        return packed_obj

    @classmethod
    def from_json(cls, data):
        validation.validate_geometry_points_fields(data)
        exterior = data[POINTS][EXTERIOR]
        if len(exterior) != 1:
            raise ValueError('"exterior" field must contain exactly one point to create "Point" object.')
        return cls(row=exterior[0][1], col=exterior[0][0])

    def crop(self, rect):
        return [self.clone()] if rect.contains_point(self) else []

    def rotate(self, rotator):
        return rotator.transform_point(self)

    def resize(self, in_size, out_size):
        new_size = image.restore_proportional_size(in_size=in_size, out_size=out_size)
        frow = new_size[0] / in_size[0]
        fcol = new_size[1] / in_size[1]
        return self._scale_frow_fcol(frow=frow, fcol=fcol)

    def scale(self, factor):
        return self._scale_frow_fcol(factor, factor)

    def _scale_frow_fcol(self, frow, fcol):
        return Point(row=round(self.row * frow), col=round(self.col * fcol))

    def translate(self, drow, dcol):
        return Point(row=(self.row + drow), col=(self.col + dcol))

    def fliplr(self, img_size):
        return Point(row=self.row, col=(img_size[1] - self.col))

    def flipud(self, img_size):
        return Point(row=(img_size[0] - self.row), col=self.col)

    def draw(self, bitmap, color, thickness=1):
        r = round(thickness / 2)  # @TODO: relation between thickness and point radius - ???
        cv2.circle(bitmap, (self.col, self.row), radius=r, color=color, thickness=cv2.FILLED)

    def draw_contour(self, bitmap, color, thickness=1):
        # @TODO: mb dummy operation for Point
        r = round((thickness + 1) / 2)
        cv2.circle(bitmap, (self.col, self.row), radius=r, color=color, thickness=cv2.FILLED)

    @property
    def area(self):
        return 0.0

    def to_bbox(self):
        raise NotImplementedError('Point object cannot produce bounding box.')


def _flip_row_col_order(coords):
    if not all(len(x) == 2 for x in coords):
        raise ValueError('Flipping row and column order values is only possible within tuples of 2 elements.')
    return [[y, x] for x, y in coords]


def _maybe_flip_row_col_order(coords, flip=False):
    return _flip_row_col_order(coords) if flip else coords


def points_to_row_col_list(points, flip_row_col_order=False):
    return _maybe_flip_row_col_order(coords=[[p.row, p.col] for p in points], flip=flip_row_col_order)


def row_col_list_to_points(data, flip_row_col_order=False, do_round=False):
    def _maybe_round(v):
        return v if not do_round else round(v)

    return [Point(row=_maybe_round(r), col=_maybe_round(c)) for r, c in
            _maybe_flip_row_col_order(data, flip=flip_row_col_order)]
