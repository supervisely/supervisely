# coding: utf-8

from copy import deepcopy

import numpy as np
from supervisely_lib.io.json import JsonSerializable
from supervisely_lib.geometry.constants import ANY_SHAPE, LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID


# @TODO: use properties instead of field if it makes sense
class Geometry(JsonSerializable):
    def __init__(self, sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        self.sly_id = sly_id
        self.labeler_login = labeler_login
        self.updated_at = updated_at
        self.created_at = created_at
        self.class_id = class_id

    def _add_creation_info(self, d):
        if self.labeler_login is not None:
            d[LABELER_LOGIN] = self.labeler_login
        if self.updated_at is not None:
            d[UPDATED_AT] = self.updated_at
        if self.created_at is not None:
            d[CREATED_AT] = self.created_at
        #@TODO: will be ignored (for compatibility) and will be used in public api
        # if self.sly_id is not None:
        #     d[ID] = self.sly_id

    @staticmethod
    def geometry_name():
        """
        :return: string with name of geometry
        """
        raise NotImplementedError()

    def crop(self, rect):
        """
        :param rect: Rectangle
        :return: list of Geometry
        """
        raise NotImplementedError()

    def relative_crop(self, rect):
        """ Crops object like "crop" method, but return results with coordinates relative to rect
        :param rect:
        :return: list of Geometry
        """
        return [geom.translate(drow=-rect.top, dcol=-rect.left) for geom in self.crop(rect)]

    def rotate(self, rotator):
        """Rotates around image center -> New Geometry
        :param rotator: ImageRotator
        :return: Geometry
        """
        raise NotImplementedError()

    def resize(self, in_size, out_size):
        """
        :param in_size: (rows, cols)
        :param out_size:
            (128, 256)
            (128, KEEP_ASPECT_RATIO)
            (KEEP_ASPECT_RATIO, 256)
        :return: Geometry
        """
        raise NotImplementedError()

    def scale(self, factor):
        """Scales around origin with a given factor.
        :param: factor (float):
        :return: Geometry
        """
        raise NotImplementedError()

    def translate(self, drow, dcol):
        """
        :param drow: int rows shift
        :param dcol: int cols shift
        :return: Geometry
        """
        raise NotImplementedError()

    def fliplr(self, img_size):
        """
        :param img_size: (rows, cols)
        :return: Geometry
        """
        raise NotImplementedError()

    def flipud(self, img_size):
        """
        :param img_size: (rows, cols)
        :return: Geometry
        """
        raise NotImplementedError()

    def _draw_bool_compatible(self, draw_fn, bitmap, color, thickness, config=None):
        if bitmap.dtype == np.bool:
            # Cannot draw on the canvas directly, create a temporary with different type.
            temp_bitmap = np.zeros(bitmap.shape[:2], dtype=np.uint8)
            draw_fn(temp_bitmap, 1, thickness=thickness, config=config)
            bitmap[temp_bitmap == 1] = color
        else:
            # Pass through the canvas without temp bitmap for efficiency.
            draw_fn(bitmap, color, thickness=thickness, config=config)

    def draw(self, bitmap, color, thickness=1, config=None):
        """
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: used only in Polyline and Point
        :param config: drawing config specific to a concrete subclass, e.g. per edge colors
        """
        self._draw_bool_compatible(self._draw_impl, bitmap, color, thickness, config)

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        """
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: used only in Polyline and Point
        """
        raise NotImplementedError()

    def draw_contour(self, bitmap, color, thickness=1, config=None):
        """ Draws the figure contour on a given bitmap canvas
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: (int)
        :param config: drawing config specific to a concrete subclass, e.g. per edge colors
        """
        self._draw_bool_compatible(self._draw_contour_impl, bitmap, color, thickness, config)

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        """ Draws the figure contour on a given bitmap canvas
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: (int)
        """
        raise NotImplementedError()

    @property
    def area(self):
        """
        :return: float
        """
        raise NotImplementedError()

    def to_bbox(self):
        """
        :return: Rectangle
        """
        raise NotImplementedError()

    def clone(self):
        return deepcopy(self)

    def validate(self, obj_class_shape, settings):
        if obj_class_shape != ANY_SHAPE:
            if self.geometry_name() != obj_class_shape:
                raise ValueError('Geometry validation error: shape names are mismatched!')

    @staticmethod
    def config_from_json(config):
        return config

    @staticmethod
    def config_to_json(config):
        return config
