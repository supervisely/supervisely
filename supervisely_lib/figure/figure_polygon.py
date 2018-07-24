# coding: utf-8

from copy import deepcopy

import cv2
import numpy as np
from shapely.geometry import mapping, Polygon as ShapelyPolygon

from ..sly_logger import logger
from .abstract_vector_figure import AbstractVectorFigure
from .rectangle import Rect


class FigurePolygon(AbstractVectorFigure):
    def _get_points_rounded(self):
        exterior, interiors = self._get_points()
        exterior = np.round(exterior).astype(int)
        interiors = [np.round(x).astype(int) for x in interiors]
        return exterior, interiors

    # for exteriors & interiors
    @classmethod
    def _normalize_ring_initially(cls, ring):
        if ring is None:
            return None
        r_np = np.asarray(ring, dtype=cls.COMMON_PTS_DTYPE)
        if len(r_np.shape) != 2:
            raise RuntimeError('Wrong points fmt (rank) in ring.')
        if r_np.shape[0] < 3:
            return None  # ok, drop it out
        if r_np.shape[1] != 2:
            raise RuntimeError('Wrong points fmt (shape[1]) in ring.')
        return r_np

    def normalize(self, img_size_wh):
        exterior, interiors = self._get_points()
        exterior = self._normalize_ring_initially(exterior)
        if exterior is None:
            return []  # ok, drop poly w/out exterior
        interiors = (self._normalize_ring_initially(x) for x in interiors)
        interiors = [x for x in interiors if x is not None]

        # @TODO: ooh
        # will not check if the polygon is valid, so any future crop may produce errors
        # also, there may be points out of roi
        self._set_points(exterior, interiors)
        return [self]

    def crop(self, rect):
        src_exterior, src_interiors = self._get_points()
        ext_bbox = Rect.from_np_points(src_exterior)
        int_bboxes = [Rect.from_np_points(x) for x in src_interiors]

        if rect.intersection(ext_bbox).is_empty:
            return []  # optimization: non-intersected
        if rect.contains(ext_bbox) and all(rect.contains(x) for x in int_bboxes):
            return [self]  # optimization: bbox contains full poly

        c_exterior, c_interiors = self._get_points()
        try:
            clipping_window_shpl = ShapelyPolygon(rect.to_np_points())
            self_shpl = ShapelyPolygon(c_exterior, holes=c_interiors)
            intersections_shpl = self_shpl.intersection(clipping_window_shpl)
            mapping_shpl = mapping(intersections_shpl)
        except Exception:
            logger.warn('Polygon cropping exception, shapely.', exc_info=False)
            raise
            # logger.warn('Polygon cropping exception, shapely.', exc_info=True)
            # # now we are dropping the res silently
            # return []

        # returns list of polygons, where polygon is iterable of rings (1st is exterior)
        # and ring is presented as tuple of (x,y)-tuples
        def shpl_to_coords_list(mp):
            if mp['type'] == 'MultiPolygon':
                return mp['coordinates']
            elif mp['type'] == 'Polygon':
                return [mp['coordinates']]
            elif mp['type'] == 'GeometryCollection':
                res = []
                for geom_obj in mp['geometries']:
                    res.extend(shpl_to_coords_list(geom_obj))
                return res
            else:
                return []

        def clipped_to_figure(intersection):
            exterior = intersection[0]
            interiors = intersection[1:]
            new_obj = deepcopy(self)
            new_obj._set_points(
                exterior=np.asarray(exterior),
                interiors=[np.asarray(interior) for interior in interiors]
            )
            return new_obj

        res_polygons_pts = shpl_to_coords_list(mapping_shpl)
        return (clipped_to_figure(x) for x in res_polygons_pts)

    def get_bbox(self):
        all_pts = self._get_points_stacked()
        rect = Rect.from_np_points(all_pts)
        return rect

    def to_bool_mask(self, shape_hw):
        exterior, interiors = self._get_points_rounded()
        bmp_to_draw = np.zeros(shape_hw, np.uint8)
        cv2.fillPoly(bmp_to_draw, pts=[exterior], color=1)
        cv2.fillPoly(bmp_to_draw, pts=interiors, color=0)
        to_contours = [interior[:, np.newaxis, :] for interior in interiors]
        cv2.drawContours(bmp_to_draw, to_contours, contourIdx=-1, color=1)
        mask_bool = bmp_to_draw.astype(bool)
        return mask_bool

    def draw(self, bitmap, color):
        self_mask = self.to_bool_mask(bitmap.shape[:2])
        bitmap[self_mask] = color

    def draw_contour(self, bitmap, color, thickness):
        exterior, interiors = self._get_points_rounded()
        poly_lines = [exterior] + interiors
        cv2.polylines(bitmap, pts=poly_lines, isClosed=True, color=color, thickness=thickness)

    # @TODO: extend possibilities, consider interiors
    # returns area of exterior figure only
    def get_area(self):
        exterior, _ = self._get_points()
        xs, ys = exterior[:, 0], exterior[:, 1]
        res = self.get_area_by_gauss_formula(xs, ys)
        return res

    def approx_dp(self, epsilon):
        exterior, interiors = self._get_points()
        exterior = self._approx_ring_dp(exterior, epsilon, closed=True)
        interiors = [self._approx_ring_dp(x, epsilon, closed=True) for x in interiors]
        self._set_points(exterior, interiors)

    def pack(self):
        # convert np arrays to lists of lists
        exterior, interiors = self._get_points()
        packed = deepcopy(self.data)
        packed['points'] = {
            'exterior': exterior.tolist(),
            'interior': [interior.tolist() for interior in interiors]
        }
        return packed

    @classmethod
    def from_packed(cls, packed_obj):
        np_points = cls.ring_to_np_points

        obj = packed_obj
        exterior = np_points(packed_obj['points'].get('exterior', []))
        if len(exterior) < 3:
            return None
        interiors = packed_obj['points'].get('interior', [])
        interiors = [np_points(ring) for ring in interiors if len(ring) >= 3]
        obj['points'] = {
            'exterior': exterior,
            'interior': interiors
        }
        return cls(obj)

    # @TODO: rewrite, validate, generalize etc
    # exterior: ring; interior: list of rings; ring: np arr of rows (x, y) or just list of pairs (x, y)
    # returns iterable
    @classmethod
    def from_np_points(cls, class_title, image_size_wh, exterior, interiors):
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
                'interior': list(interiors),
            }
        }
        temp = cls(new_data)
        res = temp.normalize(image_size_wh)
        return res

    @classmethod
    def get_area_by_gauss_formula(cls, x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
