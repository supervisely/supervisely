# coding: utf-8

from copy import deepcopy
import cv2
import numpy as np

from supervisely_lib.geometry.constants import EXTERIOR, INTERIOR, POINTS, GEOMETRY_SHAPE, GEOMETRY_TYPE
from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.point_location import PointLocation, points_to_row_col_list
from supervisely_lib.geometry.rectangle import Rectangle


class VectorGeometry(Geometry):
    '''
    This is a base class for creating and using VectorGeometry objects for Labels
    '''
    def __init__(self, exterior, interior,
                 sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        '''
        :param exterior: list of PointLocation objects
        :param interior: list of PointLocation objects
        '''
        if not (isinstance(exterior, list) and all(isinstance(p, PointLocation) for p in exterior)):
            raise TypeError('Argument "exterior" must be list of "PointLocation" objects!')

        if not isinstance(interior, list) or \
            not all(isinstance(c, list) for c in interior) or \
                not all(isinstance(p, PointLocation) for c in interior for p in c):
            raise TypeError('Argument "interior" must be list of list of "PointLocation" objects!')

        self._exterior = deepcopy(exterior)
        self._interior = deepcopy(interior)
        super().__init__(sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

    def to_json(self):
        '''
        The function from_json convert VectorGeometry class object to json format
        :return: VectorGeometry class object in json format
        '''
        packed_obj = {
            POINTS: {
                EXTERIOR: points_to_row_col_list(self._exterior, flip_row_col_order=True),
                INTERIOR: [points_to_row_col_list(i, flip_row_col_order=True) for i in self._interior]
            },
            GEOMETRY_SHAPE: self.geometry_name(),
            GEOMETRY_TYPE: self.geometry_name(),
        }
        self._add_creation_info(packed_obj)
        return packed_obj

    @property
    def exterior(self):
        return deepcopy(self._exterior)

    @property
    def exterior_np(self):
        '''
        The function exterior_np convert exterior attribute(list of PointLocation objects) to numpy array
        :return: numpy array
        '''
        return np.array(points_to_row_col_list(self._exterior), dtype=np.int64)

    @property
    def interior(self):
        return deepcopy(self._interior)

    @property
    def interior_np(self):
        '''
        The function interior_np convert interior attribute(list of PointLocation objects) to numpy array
        :return: numpy array
        '''
        return [np.array(points_to_row_col_list(i), dtype=np.int64) for i in self._interior]

    def _transform(self, transform_fn):
        result = deepcopy(self)
        result._exterior = [transform_fn(p) for p in self._exterior]
        result._interior = [[transform_fn(p) for p in i] for i in self._interior]
        return result

    def resize(self, in_size, out_size):
        '''
        Resize the current VectorGeometry to match a certain size
        :param in_size: input image size
        :param out_size: output image size
        :return: VectorGeometry class object
        '''
        return self._transform(lambda p: p.resize(in_size, out_size))

    def scale(self, factor):
        '''
        The function scale change scale of the current VectorGeometry object with a given factor
        :param factor: float scale parameter
        :return: VectorGeometry class object
        '''
        return self._transform(lambda p: p.scale(factor))

    def translate(self, drow, dcol):
        '''
        The function translate shifts the VectorGeometry object by a certain number of pixels and return the copy of the current VectorGeometry object
        :param drow: horizontal shift
        :param dcol: vertical shift
        :return: VectorGeometry class object
        '''
        return self._transform(lambda p: p.translate(drow, dcol))

    def rotate(self, rotator):
        '''
        The function rotate VectorGeometry with a given rotator(ImageRotator class object contain size of image and angle to rotate)
        :param rotator: ImageRotator class object
        :return: VectorGeometry class object
        '''
        return self._transform(lambda p: p.rotate(rotator))

    def fliplr(self, img_size):
        '''
        The function fliplr the current VectorGeometry object geometry in horizontal
        :param img_size: size of the image
        :return: VectorGeometry class object
        '''
        return self._transform(lambda p: p.fliplr(img_size))

    def flipud(self, img_size):
        '''
        The function flipud the current VectorGeometry object geometry in vertical
        :param img_size: size of the image
        :return: VectorGeometry class object
        '''
        return self._transform(lambda p: p.flipud(img_size))

    def to_bbox(self):
        '''
        The function to_bbox create Rectangle class object from current VectorGeometry class object
        :return: Rectangle class object
        '''
        exterior_np = self.exterior_np
        rows, cols = exterior_np[:, 0], exterior_np[:, 1]
        return Rectangle(top=round(min(rows).item()), left=round(min(cols).item()), bottom=round(max(rows).item()),
                         right=round(max(cols).item()))

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        """
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: used only in Polyline and Point
        """
        self._draw_contour_impl(bitmap, color, thickness, config=config)

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        """ Draws the figure contour on a given bitmap canvas
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: (int)
        """
        raise NotImplementedError()

    @staticmethod
    def _approx_ring_dp(ring, epsilon, closed):
        new_ring = cv2.approxPolyDP(ring.astype(np.int32), epsilon, closed)
        new_ring = np.squeeze(new_ring, 1)
        if len(new_ring) < 3 and closed:
            new_ring = ring.astype(np.int32)
        return new_ring

    def approx_dp(self, epsilon):
        raise NotImplementedError()
