# coding: utf-8

from copy import deepcopy

import cv2
import numpy as np
from shapely.geometry import mapping, LineString, Polygon as ShapelyPolygon

from ..sly_logger import logger
from .abstract_vector_figure import AbstractVectorFigure
from .rectangle import Rect


class FigureLine(AbstractVectorFigure):
    @classmethod
    def _normalize_line_string(cls, line_str):
        if line_str is None:
            return None
        l_np = np.asarray(line_str, dtype=cls.COMMON_PTS_DTYPE)
        if len(l_np.shape) != 2:
            raise RuntimeError('Wrong points fmt (rank) in line string.')
        if l_np.shape[0] < 2:
            return None  # ok, drop it out
        if l_np.shape[1] != 2:
            raise RuntimeError('Wrong points fmt (shape[1]) in line string.')
        return l_np

    def normalize(self, img_size_wh):
        exterior, _ = self._get_points()
        exterior = self._normalize_line_string(exterior)
        if exterior is None:
            return []  # ok, drop empty line
        self._set_points(exterior)

        crop_r = Rect.from_size(img_size_wh)
        res = self.crop(crop_r)
        return res

    def crop(self, rect):
        src_exterior, _ = self._get_points()
        ext_bbox = Rect.from_np_points(src_exterior)
        if rect.intersection(ext_bbox).is_empty and (not ext_bbox.is_empty):
            return []  # optimization: non-intersected
        if rect.contains(ext_bbox):
            return [self]  # optimization: bbox contains full poly

        try:
            clipping_window_shpl = ShapelyPolygon(rect.to_np_points())
            self_shpl = LineString(src_exterior)
            intersections_shpl = self_shpl.intersection(clipping_window_shpl)
            mapping_shpl = mapping(intersections_shpl)
        except Exception:
            logger.warn('Line cropping exception, shapely.', exc_info=False)
            raise

        # returns list of polygons, where polygon is iterable of rings (1st is exterior)
        # and ring is presented as tuple of (x,y)-tuples
        def shpl_to_coords_list(mp):
            if mp['type'] == 'MultiLineString':
                return mp['coordinates']
            elif mp['type'] == 'LineString':
                return [mp['coordinates']]
            elif mp['type'] == 'GeometryCollection':
                res = []
                for geom_obj in mp['geometries']:
                    res.extend(shpl_to_coords_list(geom_obj))
                return res
            else:
                return []

        def clipped_to_figure(intersection):
            exterior = intersection
            new_obj = deepcopy(self)
            new_obj._set_points(exterior=np.asarray(exterior))
            return new_obj

        res_lines_pts = shpl_to_coords_list(mapping_shpl)

        # tiny hack to combine consequtive segments
        lines_combined = []
        for simple_l in res_lines_pts:
            if len(lines_combined) > 0:
                prev = lines_combined[-1]
                if prev[-1] == simple_l[0]:
                    lines_combined[-1] = list(prev) + list(simple_l[1:])
                    continue
            lines_combined.append(simple_l)

        return (clipped_to_figure(x) for x in lines_combined)

    def get_bbox(self):
        exterior, _ = self._get_points()
        rect = Rect.from_np_points(exterior)
        return rect

    def to_bool_mask(self, shape_hw):
        bmp_to_draw = np.zeros(shape_hw, np.uint8)
        self.draw(bmp_to_draw, color=1)
        mask_bool = bmp_to_draw.astype(bool)
        return mask_bool

    def draw(self, bitmap, color):
        self.draw_contour(bitmap, color, thickness=1)  # due to cv2

    def draw_contour(self, bitmap, color, thickness):
        exterior, _ = self._get_points()
        exterior = np.round(exterior).astype(int)
        cv2.polylines(bitmap, pts=[exterior], isClosed=False, color=color, thickness=thickness)

    def get_area(self):
        return 0

    def approx_dp(self, epsilon):
        exterior, _ = self._get_points()
        exterior = self._approx_ring_dp(exterior, epsilon, closed=False)
        self._set_points(exterior)

    @classmethod
    def from_packed(cls, packed_obj):
        obj = packed_obj
        exterior = cls.ring_to_np_points(packed_obj['points'].get('exterior', []))
        if len(exterior) < 2:
            return None
        obj['points'] = {
            'exterior': exterior,
            'interior': []
        }
        return cls(obj)

    # @TODO: rewrite, validate, generalize etc
    # exterior: line str, i.e. np arr of rows (x, y) or just list of pairs (x, y)
    # returns iterable
    @classmethod
    def from_np_points(cls, class_title, image_size_wh, exterior):
        new_data = {
            'bitmap': {
                'origin': [],
                'np': [],
            },
            'type': 'polygon',
            'classTitle': class_title,
            'description': '',
            'tags': [],
            'points': {
                'exterior': list(exterior),
                'interior': []
            }
        }
        temp = cls(new_data)
        res = temp.normalize(image_size_wh)
        return res
