# coding: utf-8

import cv2
import numpy as np

from .abstract_vector_figure import AbstractVectorFigure
from .rectangle import Rect


class FigureRectangle(AbstractVectorFigure):
    @classmethod
    def _exterior_from_rect(cls, r):
        res = cls.ring_to_np_points((r.point0, r.point1))
        return res

    def normalize(self, img_size_wh):
        crop_r = Rect.from_size(img_size_wh)
        res = self.crop(crop_r)  # must store points in correct fmt
        return res

    def crop(self, rect):
        self_r = self.get_bbox()
        clipped = rect.intersection(self_r)
        if clipped.is_empty:
            return []
        self._set_points(self._exterior_from_rect(clipped))
        return [self]

    def rotate(self, _):
        raise RuntimeError('Unable to rotate Rectangle figure.')

    def get_bbox(self):
        pts = self.data['points']['exterior']
        rect = Rect.from_np_points(pts[:2])
        return rect

    def to_bool_mask(self, shape_hw):
        bmp_to_draw = np.zeros(shape_hw, np.uint8)
        self.draw_contour(bmp_to_draw, color=1, thickness=cv2.FILLED)  # due to cv2
        mask_bool = bmp_to_draw.astype(bool)
        return mask_bool

    def draw(self, bitmap, color):
        self.draw_contour(bitmap, color, thickness=cv2.FILLED)  # due to cv2

    def draw_contour(self, bitmap, color, thickness):
        self_r = self.get_bbox().round()
        cv2.rectangle(bitmap, pt1=self_r.point0, pt2=self_r.point1, color=color, thickness=thickness)

    def get_area(self):
        res = self.get_bbox().area
        return res

    @classmethod
    def from_packed(cls, packed_obj):
        obj = packed_obj
        exterior = cls.ring_to_np_points(packed_obj['points'].get('exterior', []))
        if len(exterior) < 2:
            return None
        obj['points'] = {
            'exterior': exterior[:2, :],
            'interior': []
        }
        return cls(obj)

    # @TODO: rewrite, validate, generalize etc
    # returns iterable
    @classmethod
    def from_rect(cls, class_title, image_size_wh, rect):
        new_data = {
            'bitmap': {
                'origin': [],
                'np': [],
            },
            'type': 'rectangle',
            'classTitle': class_title,
            'description': '',
            'tags': [],
            'points': {
                'interior': [],
                'exterior': cls._exterior_from_rect(rect)
            },
        }
        temp = cls(new_data)
        res = temp.normalize(image_size_wh)
        return res
