# coding: utf-8

import cv2
import numpy as np

from shapely.geometry import mapping, LineString, Polygon as ShapelyPolygon
from supervisely_lib.geometry.conversions import shapely_figure_to_coords_list
from supervisely_lib.geometry.point_location import row_col_list_to_points
from supervisely_lib.geometry.vector_geometry import VectorGeometry
from supervisely_lib.geometry.constants import EXTERIOR, POINTS
from supervisely_lib.geometry import validation
from supervisely_lib import logger


class Polyline(VectorGeometry):
    @staticmethod
    def geometry_name():
        return 'line'

    def __init__(self, exterior):
        """
        :param exterior: [PointLocation]
        """
        if len(exterior) < 2:
            raise ValueError('"{}" field must contain at least two points to create "Polyline" object.'
                             .format(EXTERIOR))

        super().__init__(exterior=exterior, interior=[])

    @classmethod
    def from_json(cls, data):
        validation.validate_geometry_points_fields(data)
        return cls(exterior=row_col_list_to_points(data[POINTS][EXTERIOR], flip_row_col_order=True))

    def crop(self, rect):
        try:
            clipping_window = [[rect.left, rect.top], [rect.right, rect.top],
                               [rect.right, rect.bottom], [rect.left, rect.bottom]]
            clipping_window_shpl = ShapelyPolygon(clipping_window)

            exterior = self.exterior_np[:, ::-1]
            intersections_polygon = LineString(exterior).intersection(clipping_window_shpl)
            mapping_shpl = mapping(intersections_polygon)
        except Exception:
            logger.warn('Line cropping exception, shapely.', exc_info=False)
            raise

        res_lines_pts = shapely_figure_to_coords_list(mapping_shpl)

        # tiny hack to combine consecutive segments
        lines_combined = []
        for simple_l in res_lines_pts:
            if len(lines_combined) > 0:
                prev = lines_combined[-1]
                if prev[-1] == simple_l[0]:
                    lines_combined[-1] = list(prev) + list(simple_l[1:])
                    continue
            lines_combined.append(simple_l)

        return [Polyline(row_col_list_to_points(line)) for line in lines_combined]

    def _draw_impl(self, bitmap: np.ndarray, color, thickness=1, config=None):
        self._draw_contour_impl(bitmap, color, thickness, config=config)

    def _draw_contour_impl(self, bitmap: np.ndarray, color, thickness=1, config=None):
        exterior = self.exterior_np[:, ::-1]
        cv2.polylines(bitmap, pts=[exterior], isClosed=False, color=color, thickness=thickness)

    @property
    def area(self):
        return 0.0

    def approx_dp(self, epsilon):
        exterior_np = self._approx_ring_dp(self.exterior_np, epsilon, closed=True).tolist()
        exterior = row_col_list_to_points(exterior_np, do_round=True)
        return Polyline(exterior)
