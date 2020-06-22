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
        '''
        The function to_json convert PointLocation class object to json format
        :return: PointLocation in json format
        '''
        packed_obj = {
            POINTS: {
                EXTERIOR: [[self.col, self.row]],
                INTERIOR: []
            }
        }
        return packed_obj

    @classmethod
    def from_json(cls, data):
        '''
        The function from_json convert PointLocation from json format to PointLocation class object.
        :param data: input PointLocation in json format
        :return: PointLocation class object
        '''
        validation.validate_geometry_points_fields(data)
        exterior = data[POINTS][EXTERIOR]
        if len(exterior) != 1:
            raise ValueError('"exterior" field must contain exactly one point to create "PointLocation" object.')
        return cls(row=exterior[0][1], col=exterior[0][0])

    def scale(self, factor):
        '''
        The function scale calculates new parameters of point after scaling image
        :param factor: float scale parameter
        :return: PointLocation class object
        '''
        return self.scale_frow_fcol(factor, factor)

    def scale_frow_fcol(self, frow, fcol):
        '''
        The function scale_frow_fcol calculates new parameters of point after scaling with given parameters in horizontal and vertical
        :param frow: float scale parameter
        :param fcol: float scale parameter
        :return: PointLocation class object
        '''
        return PointLocation(row=round(self.row * frow), col=round(self.col * fcol))

    def translate(self, drow, dcol):
        '''
        The function translate calculates new parameters of point after shifts it by a certain number of pixels
        :param drow: horizontal shift
        :param dcol: vertical shift
        :return: PointLocation class object
        '''
        return PointLocation(row=(self.row + drow), col=(self.col + dcol))

    def rotate(self, rotator):
        '''
        The function rotate calculates new parameters of point after rotating
        :param rotator: ImageRotator class object
        :return: PointLocation class object
        '''
        return rotator.transform_point(self)

    def resize(self, in_size, out_size):
        '''
        The function resize calculates new parameters of point after resizing image
        :param in_size: input image size
        :param out_size: output image size
        :return: PointLocation class object
        '''
        new_size = sly_image.restore_proportional_size(in_size=in_size, out_size=out_size)
        frow = new_size[0] / in_size[0]
        fcol = new_size[1] / in_size[1]
        return self.scale_frow_fcol(frow=frow, fcol=fcol)

    def fliplr(self, img_size):
        '''
        The function fliplr calculates new parameters of point after fliping image in horizontal
        :param img_size: image size
        :return: PointLocation class object
        '''
        return PointLocation(row=self.row, col=(img_size[1] - self.col))

    def flipud(self, img_size):
        '''
        The function flipud calculates new parameters of point after fliping image in vertical
        :param img_size: image size
        :return: PointLocation class object
        '''
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
    '''
    The function points_to_row_col_list convert list of points(PointLocation objects) to list of coords, flip row col coords if flip_row_col_order==True
    :param points: list of points(PointLocation class objects)
    :param flip_row_col_order: bool param for flipping coords
    :return: list of coords
    '''
    return _maybe_flip_row_col_order(coords=[[p.row, p.col] for p in points], flip=flip_row_col_order)


def row_col_list_to_points(data, flip_row_col_order=False, do_round=False):
    '''
    The function row_col_list_to_points convertlist of coords to list of points(PointLocation objects), flip row col coords if flip_row_col_order==True, round PointLocation params if do_round==True
    :param data: list of coords
    :param flip_row_col_order: bool param for flipping coords
    :param do_round: bool param to round coords
    :return: list of points(PointLocation class objects)
    '''
    def _maybe_round(v):
        return v if not do_round else round(v)

    return [PointLocation(row=_maybe_round(r), col=_maybe_round(c)) for r, c in
            _maybe_flip_row_col_order(data, flip=flip_row_col_order)]
