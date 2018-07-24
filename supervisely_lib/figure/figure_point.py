# coding: utf-8

import cv2
import numpy as np

from .abstract_vector_figure import AbstractVectorFigure
from .rectangle import Rect


class FigurePoint(AbstractVectorFigure):
    @classmethod
    def _exterior_from_pt(cls, xy_tuple):
        res = cls.ring_to_np_points((xy_tuple, ))
        return res

    def normalize(self, img_size_wh):
        self._set_points(self._exterior_from_pt(self.get_point()))  # drop excess values
        crop_r = Rect.from_size(img_size_wh)
        res = self.crop(crop_r)  # must store points in correct fmt
        return res

    def crop(self, rect):
        self_pt = self.get_point()
        save = rect.contains_point(self_pt)
        if not save:
            return []
        return [self]

    def get_point(self):
        pts = self.data['points']['exterior']
        pt = (pts[0, 0], pts[0, 1])
        return pt

    def get_bbox(self):
        pt = self.get_point()
        rect = Rect(pt[0], pt[1], pt[0], pt[1])
        return rect

    def to_bool_mask(self, shape_hw):
        mask_bool = np.zeros(shape_hw, np.bool)
        self.draw(mask_bool, True)
        return mask_bool

    def draw(self, bitmap, color):
        pt = self.get_point()
        pt = [int(np.round(x)) for x in pt]
        shape = bitmap.shape
        mask_r = Rect.from_size((shape[1] - 1, shape[0] - 1))  # in array
        if mask_r.contains_point(pt):
            bitmap[pt[1], pt[0]] = color

    def draw_contour(self, bitmap, color, thickness):
        pt = self.get_point()
        pt = [int(np.round(x)) for x in pt]
        r = int((thickness + 1) / 2)
        cv2.circle(bitmap, tuple(pt), radius=r, color=color, thickness=cv2.FILLED)

    def get_area(self):
        return 0

    @classmethod
    def from_packed(cls, packed_obj):
        obj = packed_obj
        exterior = cls.ring_to_np_points(packed_obj['points'].get('exterior', []))
        if len(exterior) == 0:
            return None
        obj['points'] = {
            'exterior': exterior[:1, :],
            'interior': []
        }
        return cls(obj)

    # @TODO: rewrite, validate, generalize etc
    # returns single obj
    @classmethod
    def from_pt(cls, class_title, xy_tuple):
        new_data = {
            'bitmap': {
                'origin': [],
                'np': [],
            },
            'type': 'point',
            'classTitle': class_title,
            'description': '',
            'tags': [],
            'points': {
                'interior': [],
                'exterior': cls._exterior_from_pt(xy_tuple)
            },
        }
        res = cls(new_data)
        return res
