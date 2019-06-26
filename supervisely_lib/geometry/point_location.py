# coding: utf-8

from copy import deepcopy

from supervisely_lib.io.json import JsonSerializable
from supervisely_lib.imaging import image as sly_image
from supervisely_lib.geometry import validation
from supervisely_lib.geometry.constants import EXTERIOR, INTERIOR, POINTS
from supervisely_lib._utils import unwrap_if_numpy


class PointLocation(JsonSerializable):
    def __init__(self, row, col):
        """
        Create simple point in (row, col) position. Float-type coordinates will be deprecated soon.
        """
        self._row = round(unwrap_if_numpy(row))
        self._col = round(unwrap_if_numpy(col))

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
            raise ValueError('"exterior" field must contain exactly one point to create "PointLocation" object.')
        return cls(row=exterior[0][1], col=exterior[0][0])

    def scale(self, factor):
        return self.scale_frow_fcol(factor, factor)

    def scale_frow_fcol(self, frow, fcol):
        return PointLocation(row=round(self.row * frow), col=round(self.col * fcol))

    def translate(self, drow, dcol):
        return PointLocation(row=(self.row + drow), col=(self.col + dcol))

    def rotate(self, rotator):
        return rotator.transform_point(self)

    def resize(self, in_size, out_size):
        new_size = sly_image.restore_proportional_size(in_size=in_size, out_size=out_size)
        frow = new_size[0] / in_size[0]
        fcol = new_size[1] / in_size[1]
        return self.scale_frow_fcol(frow=frow, fcol=fcol)

    def fliplr(self, img_size):
        return PointLocation(row=self.row, col=(img_size[1] - self.col))

    def flipud(self, img_size):
        return PointLocation(row=(img_size[0] - self.row), col=self.col)

    def clone(self):
        return deepcopy(self)


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

    return [PointLocation(row=_maybe_round(r), col=_maybe_round(c)) for r, c in
            _maybe_flip_row_col_order(data, flip=flip_row_col_order)]
