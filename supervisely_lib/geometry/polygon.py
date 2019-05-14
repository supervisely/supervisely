# coding: utf-8

import cv2
import numpy as np

from shapely.geometry import mapping, Polygon as ShapelyPolygon

from supervisely_lib.geometry.conversions import shapely_figure_to_coords_list
from supervisely_lib.geometry.point_location import row_col_list_to_points, points_to_row_col_list
from supervisely_lib.geometry.vector_geometry import VectorGeometry
from supervisely_lib.geometry.constants import EXTERIOR, INTERIOR, POINTS
from supervisely_lib.geometry import validation
from supervisely_lib.sly_logger import logger


class Polygon(VectorGeometry):
    @staticmethod
    def geometry_name():
        return 'polygon'

    def __init__(self, exterior, interior):
        if len(exterior) < 3:
            raise ValueError('"{}" field must contain at least 3 points to create "Polygon" object.'.format(EXTERIOR))
        if any(len(element) < 3 for element in interior):
            raise ValueError('"{}" element must contain at least 3 points.'.format(INTERIOR))

        super().__init__(exterior, interior)

    @classmethod
    def from_json(cls, data):
        validation.validate_geometry_points_fields(data)
        return cls(exterior=row_col_list_to_points(data[POINTS][EXTERIOR], flip_row_col_order=True),
                   interior=[row_col_list_to_points(i, flip_row_col_order=True) for i in data[POINTS][INTERIOR]])

    def crop(self, rect):
        try:
            clipping_window_shpl = ShapelyPolygon(points_to_row_col_list(rect.corners))
            self_shpl = ShapelyPolygon(self.exterior_np, holes=self.interior_np)
            intersections_shpl = self_shpl.buffer(0).intersection(clipping_window_shpl)
            mapping_shpl = mapping(intersections_shpl)
        except Exception:
            logger.warn('Polygon cropping exception, shapely.', exc_info=False)
            raise

        intersections = shapely_figure_to_coords_list(mapping_shpl)

        # Check for bad cropping cases (e.g. empty points list)
        out_polygons = []
        for intersection in intersections:
            if isinstance(intersection, list) and len(intersection) > 0 and len(intersection[0]) >= 3:
                exterior = row_col_list_to_points(intersection[0], do_round=True)
                interiors = []
                for interior_contour in intersection[1:]:
                    if len(interior_contour) > 2:
                        interiors.append(row_col_list_to_points(interior_contour, do_round=True))
                out_polygons.append(Polygon(exterior, interiors))
        return out_polygons

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        exterior = self.exterior_np[:, ::-1]
        interior = [x[:, ::-1] for x in self.interior_np]
        bmp_to_draw = np.zeros(bitmap.shape[:2], np.uint8)
        cv2.fillPoly(bmp_to_draw, pts=[exterior], color=1)
        cv2.fillPoly(bmp_to_draw, pts=interior, color=0)
        bool_mask = bmp_to_draw.astype(bool)
        bitmap[bool_mask] = color

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        exterior = self.exterior_np[:, ::-1]
        interior = [x[:, ::-1] for x in self.interior_np]

        poly_lines = [exterior] + interior
        cv2.polylines(bitmap, pts=poly_lines, isClosed=True, color=color, thickness=thickness)

    # @TODO: extend possibilities, consider interior
    # returns area of exterior figure only
    @property
    def area(self):
        exterior = self.exterior_np
        return self._get_area_by_gauss_formula(exterior[:, 0], exterior[:, 1])

    @staticmethod
    def _get_area_by_gauss_formula(rows, cols):
        return 0.5 * np.abs(np.dot(rows, np.roll(cols, 1)) - np.dot(cols, np.roll(rows, 1)))

    def approx_dp(self, epsilon):
        exterior_np = self._approx_ring_dp(self.exterior_np, epsilon, closed=True).tolist()
        interior_np = [self._approx_ring_dp(x, epsilon, closed=True).tolist() for x in self.interior_np]
        exterior = row_col_list_to_points(exterior_np, do_round=True)
        interior = [row_col_list_to_points(x, do_round=True) for x in interior_np]
        return Polygon(exterior, interior)
