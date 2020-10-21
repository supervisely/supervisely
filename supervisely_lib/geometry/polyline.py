# coding: utf-8

import cv2
import numpy as np

from shapely.geometry import mapping, LineString, Polygon as ShapelyPolygon
from supervisely_lib.geometry.conversions import shapely_figure_to_coords_list
from supervisely_lib.geometry.point_location import row_col_list_to_points
from supervisely_lib.geometry.vector_geometry import VectorGeometry
from supervisely_lib.geometry.constants import EXTERIOR, POINTS, LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID, CLASS_ID
from supervisely_lib.geometry import validation
from supervisely_lib import logger


class Polyline(VectorGeometry):
    '''
    This is a class for creating and using Polyline objects for Labels
    '''
    @staticmethod
    def geometry_name():
        return 'line'

    def __init__(self, exterior,
                 sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        """
        :param exterior: list of PointLocation objects
        """
        if len(exterior) < 2:
            raise ValueError('"{}" field must contain at least two points to create "Polyline" object.'
                             .format(EXTERIOR))

        super().__init__(exterior=exterior, interior=[], sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at,
                         created_at=created_at)

    @classmethod
    def from_json(cls, data):
        '''
        The function from_json convert Polyline from json format to Poligon class object. If json format is not correct it generate exception error.
        :param data: input Polyline in json format
        :return: Polyline class object
        '''
        validation.validate_geometry_points_fields(data)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return cls(exterior=row_col_list_to_points(data[POINTS][EXTERIOR], flip_row_col_order=True),
                   sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

    def crop(self, rect):
        '''
        Crop the current Polyline with a given rectangle, if polyline cat't be cropped it generate exception error
        :param rect: Rectangle class object
        :return: list of Polyline class objects
        '''
        try:
            clipping_window = [[rect.top, rect.left], [rect.top, rect.right],
                               [rect.bottom, rect.right], [rect.bottom, rect.left]]
            clipping_window_shpl = ShapelyPolygon(clipping_window)

            exterior = self.exterior_np
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
        '''
        The function approx_dp approximates a polygonal curve with the specified precision
        :param epsilon: Parameter specifying the approximation accuracy. This is the maximum distance between the original curve and its approximation
        :return: Polyline class object
        '''
        exterior_np = self._approx_ring_dp(self.exterior_np, epsilon, closed=True).tolist()
        exterior = row_col_list_to_points(exterior_np, do_round=True)
        return Polyline(exterior)

    @classmethod
    def allowed_transforms(cls):
        from supervisely_lib.geometry.any_geometry import AnyGeometry
        from supervisely_lib.geometry.rectangle import Rectangle
        from supervisely_lib.geometry.bitmap import Bitmap
        from supervisely_lib.geometry.polygon import Polygon
        return [AnyGeometry, Rectangle, Bitmap, Polygon]