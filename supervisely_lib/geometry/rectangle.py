# coding: utf-8

import cv2
import numpy as np

from supervisely_lib.geometry.constants import EXTERIOR, INTERIOR, POINTS, LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID, CLASS_ID
from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.point_location import PointLocation, points_to_row_col_list
from supervisely_lib.geometry import validation


# @TODO: validation
class Rectangle(Geometry):
    '''
    This is a class for creating and using Rectangle objects for Labels
    '''
    @staticmethod
    def geometry_name():
        return 'rectangle'

    def __init__(self, top, left, bottom, right,
                 sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        """
        Float-type coordinates will be deprecated soon.
        Args:
            top: minimal vertical value
            left: minimal horizontal value
            bottom: maximal vertical value
            right: maximal horizontal value
        """

        if top > bottom:
            raise ValueError('Rectangle "top" argument must have less or equal value then "bottom"!')

        if left > right:
            raise ValueError('Rectangle "left" argument must have less or equal value then "right"!')

        super().__init__(sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at,
                         created_at=created_at)

        self._points = [PointLocation(row=top, col=left), PointLocation(row=bottom, col=right)]

    """
    Implementation of all methods from Geometry
    """

    def to_json(self):
        '''
        The function to_json convert Rectangle class object to json format
        :return: Rectangle in json format
        '''
        packed_obj = {
            POINTS: {
                EXTERIOR: points_to_row_col_list(self._points, flip_row_col_order=True),
                INTERIOR: []
            }
        }
        self._add_creation_info(packed_obj)
        return packed_obj

    @classmethod
    def from_json(cls, data):
        '''
        The function from_json convert Rectangle from json format to Rectangle class object. If json format is not correct it generate exception error.
        :param data: input Rectangle in json format
        :return: Rectangle class object
        '''
        validation.validate_geometry_points_fields(data)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)

        exterior = data[POINTS][EXTERIOR]
        if len(exterior) != 2:
            raise ValueError('"exterior" field must contain exactly two points to create Rectangle object.')
        [top, bottom] = sorted([exterior[0][1], exterior[1][1]])
        [left, right] = sorted([exterior[0][0], exterior[1][0]])
        return cls(top=top, left=left, bottom=bottom, right=right,
                   sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

    def crop(self, other):
        '''
        Crop the current Rectangle with a given rectangle
        :param other: Rectangle class object
        :return: list with Rectangle class object, if Rectangle class object not intersect with given rectangle, empty list will be returned
        '''
        top = max(self.top, other.top)
        left = max(self.left, other.left)
        bottom = min(self.bottom, other.bottom)
        right = min(self.right, other.right)
        is_valid = (bottom >= top) and (left <= right)
        return [Rectangle(top=top, left=left, bottom=bottom, right=right)] if is_valid else []

    def _transform(self, transform_fn):
        transformed_corners = [transform_fn(p) for p in self.corners]
        rows, cols = zip(*points_to_row_col_list(transformed_corners))
        return Rectangle(top=round(min(rows)), left=round(min(cols)), bottom=round(max(rows)), right=round(max(cols)))

    @property
    def corners(self):
        '''
        :return: list of coners(PointLocation class objects) of Rectangle
        '''
        return [PointLocation(row=self.top, col=self.left), PointLocation(row=self.top, col=self.right),
                PointLocation(row=self.bottom, col=self.right), PointLocation(row=self.bottom, col=self.left)]

    def rotate(self, rotator):
        '''
        The function rotate Rectangle with a given rotator(ImageRotator class object contain size of image and angle to rotate)
        :param rotator: ImageRotator class object
        :return: Rectangle class object
        '''
        return self._transform(lambda p: rotator.transform_point(p))

    def resize(self, in_size, out_size):
        '''
        Resize the current Rectangle to match a certain size
        :param in_size: input image size
        :param out_size: output image size
        :return: Rectangle class object
        '''
        return self._transform(lambda p: p.resize(in_size, out_size))

    def scale(self, factor):
        '''
        The function scale change scale of the current Rectangle object with a given factor
        :param factor: float scale parameter
        :return: Rectangle class object
        '''
        return self._transform(lambda p: p.scale(factor))

    def translate(self, drow, dcol):
        '''
        The function translate shifts the rectangle by a certain number of pixels and return the copy of the current Rectangle object
        :param drow: horizontal shift
        :param dcol: vertical shift
        :return: Rectangle class object
        '''
        return self._transform(lambda p: p.translate(drow, dcol))

    def fliplr(self, img_size):
        '''
        The function fliplr the current Rectangle object geometry in horizontal
        :param img_size: size of the image
        :return: Rectangle class object
        '''
        img_width = img_size[1]
        return Rectangle(top=self.top, left=(img_width - self.right), bottom=self.bottom, right=(img_width - self.left))

    def flipud(self, img_size):
        '''
        The function flipud the current Rectangle object geometry in vertical
        :param img_size: size of the image
        :return: Rectangle class object
        '''
        img_height = img_size[0]
        return Rectangle(top=(img_height - self.bottom), left=self.left, bottom=(img_height - self.top),
                         right=self.right)

    def _draw_impl(self, bitmap: np.ndarray, color, thickness=1, config=None):
        self._draw_contour_impl(bitmap, color, thickness=cv2.FILLED, config=config)  # due to cv2

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        cv2.rectangle(bitmap, pt1=(self.left, self.top), pt2=(self.right, self.bottom), color=color,
                      thickness=thickness)

    def to_bbox(self):
        '''
        :return: copy of current Rectangle class object
        '''
        return self.clone()

    @property
    def area(self):
        '''
        :return: area of current Rectangle object
        '''
        return float(self.width * self.height)

    @classmethod
    def from_array(cls, arr):
        '''
        The function from_array create Rectangle object with given array shape
        :param arr: numpy array
        :return: Rectangle class object
        '''
        return cls(top=0, left=0, bottom=arr.shape[0] - 1, right=arr.shape[1] - 1)

    # TODO re-evaluate whether we need this, looks trivial.
    @classmethod
    def from_size(cls, size: tuple):
        '''
        The function from_size create Rectangle object with given size shape
        :param size: tuple of integers
        :return: Rectangle class object
        '''
        return cls(0, 0, size[0] - 1, size[1] - 1)

    @classmethod
    def from_geometries_list(cls, geometries):
        '''
        The function from_geometries_list create Rectangle object from given geometry object
        :param geometries: list of geometry type objects(Point, Polygon, PolyLine, Bitmap etc.)
        :return: Rectangle class object
        '''
        bboxes = [g.to_bbox() for g in geometries]
        top = min(bbox.top for bbox in bboxes)
        left = min(bbox.left for bbox in bboxes)
        bottom = max(bbox.bottom for bbox in bboxes)
        right = max(bbox.right for bbox in bboxes)
        return cls(top=top, left=left, bottom=bottom, right=right)

    @property
    def left(self):
        return self._points[0].col

    @property
    def right(self):
        return self._points[1].col

    @property
    def top(self):
        return self._points[0].row

    @property
    def bottom(self):
        return self._points[1].row

    @property
    def center(self):
        '''
        :return: center of rectangle(PointLocation class obgect)
        '''
        return PointLocation(row=(self.top + self.bottom) // 2, col=(self.left + self.right) // 2)

    @property
    def width(self):
        '''
        :return: width of rectangle(int)
        '''
        return self.right - self.left + 1

    @property
    def height(self):
        '''
        :return: height of rectangle(int)
        '''
        return self.bottom - self.top + 1

    def contains(self, rect):
        '''
        The function contains checks if Rectangle class object contains a given rectangle
        :param rect: Rectangle class object
        :return: bool
        '''
        return (self.left <= rect.left and
                self.right >= rect.right and
                self.top <= rect.top and
                self.bottom >= rect.bottom)

    def contains_point_location(self, pt: PointLocation):
        '''
        The function contains_point_location checks if Rectangle class object contains a given point
        :param pt: PointLocation class object
        :return: bool
        '''
        return (self.left <= pt.col <= self.right) and (self.top <= pt.row <= self.bottom)

    def to_size(self):
        '''
        :return: height and width of rectangle(int)
        '''
        return self.height, self.width

    def get_cropped_numpy_slice(self, data: np.ndarray) -> np.ndarray:
        '''
        The function get_cropped_numpy_slice checks slice of given numpy array with Rectangle parameters
        :param data: numpy array
        :return: numpy array
        '''
        return data[self.top:(self.bottom+1), self.left:(self.right+1), ...]

    def intersects_with(self, rect):
        if self.left > rect.right or self.right < rect.left:
            return False
        if self.top > rect.bottom or self.bottom < rect.top:
            return False
        return True

    @classmethod
    def allowed_transforms(cls):
        from supervisely_lib.geometry.any_geometry import AnyGeometry
        from supervisely_lib.geometry.bitmap import Bitmap
        from supervisely_lib.geometry.polygon import Polygon
        return [AnyGeometry, Bitmap, Polygon]
