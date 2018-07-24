# coding: utf-8

import json

from .aux import not_implemeted_method


# abstract base class for figures
# packed_obj is native python dict which has been read from json or may be jsonized
class AbstractFigure:
    # obj is dict with required fields
    def __init__(self, obj):
        self.data = obj

    def __str__(self):
        return json.dumps(self.data, sort_keys=True, indent=4)

    # returns figure (w/out validation) or None; owns packed_obj
    @classmethod
    def from_packed(cls, packed_obj):
        not_implemeted_method()

    # returns packed obj
    def pack(self):
        not_implemeted_method()

    # checks if figure is valid & transforms if required; returns iterable
    def normalize(self, img_size_wh):
        not_implemeted_method()

    # expects normalized rect; returns iterable
    # note: it shall not shift objects
    def crop(self, rect):
        not_implemeted_method()

    # rotator instance provides methods to operate; operates in-place
    def rotate(self, rotator):
        not_implemeted_method()

    # resizer instance provides methods to operate;
    def resize(self, resizer):
        not_implemeted_method()

    # delta as integer (dx, dy); operates in-place
    def shift(self, delta):
        not_implemeted_method()

    # Horizontal if is_horiz == True, else vertical; operates in-place
    def flip(self, is_horiz, img_shape):
        not_implemeted_method()

    # returns Rect
    def get_bbox(self):
        not_implemeted_method()

    # returns 2d bool mask
    def to_bool_mask(self, shape_hw):
        not_implemeted_method()

    # draws on 1d or 3d uint8 bitmap with given color; operates in-place
    # bitmap must have enough size; color must be int or 3 ints corresp with bitmap
    def draw(self, bitmap, color):
        not_implemeted_method()

    # acts like draw by accepts thickness
    def draw_contour(self, bitmap, color, thickness):
        not_implemeted_method()

    def get_area(self):
        not_implemeted_method()

    @property
    def class_title(self):
        return self.data['classTitle']

    @class_title.setter
    def class_title(self, value):
        self.data['classTitle'] = value
