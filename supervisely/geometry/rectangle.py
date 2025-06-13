# coding: utf-8

# docs
from __future__ import annotations

from copy import deepcopy
from math import ceil, floor
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

import supervisely as sly
from supervisely.geometry import validation
from supervisely.geometry.constants import (
    CLASS_ID,
    CREATED_AT,
    EXTERIOR,
    ID,
    INTERIOR,
    LABELER_LOGIN,
    POINTS,
    UPDATED_AT,
)
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.point_location import PointLocation, points_to_row_col_list


class Rectangle(Geometry):
    """
    Rectangle geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`Rectangle<Rectangle>` class object is immutable.

    :param top: Minimal vertical value of Rectangle object.
    :type top: int or float
    :param left: Minimal horizontal value of Rectangle object.
    :type left: int or float
    :param bottom: Maximal vertical value of Rectangle object.
    :type bottom: int or float
    :param right: Maximal vertical value of Rectangle object.
    :type right: int or float
    :param sly_id: Rectangle ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which Rectangle belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created Rectangle.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when Rectangle was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when Rectangle was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :raises: :class:`ValueError`. Rectangle top argument must have less or equal value then bottom, left argument must have less or equal value then right

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        top = 100
        left = 100
        bottom = 700
        right = 900
        figure = sly.Rectangle(top, left, bottom, right)
    """

    @staticmethod
    def geometry_name():
        """ """
        return "rectangle"

    def __init__(
        self,
        top: int,
        left: int,
        bottom: int,
        right: int,
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[int] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):

        if top > bottom:
            raise ValueError(
                'Rectangle "top" argument must have less or equal value then "bottom"!'
            )

        if left > right:
            raise ValueError(
                'Rectangle "left" argument must have less or equal value then "right"!'
            )

        super().__init__(
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

        self._points = [
            PointLocation(row=top, col=left),
            PointLocation(row=bottom, col=right),
        ]

    def to_json(self) -> Dict:
        """
        Convert the Rectangle to a json dict. Read more about `Supervisely format <https://docs.supervisely.com/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            figure_json = figure.to_json()
            print(figure_json)
            # Output: {
            #    "points": {
            #        "exterior": [
            #            [100, 100],
            #            [900, 700]
            #        ],
            #        "interior": []
            #    }
            # }
        """
        packed_obj = {
            POINTS: {
                EXTERIOR: points_to_row_col_list(self._points, flip_row_col_order=True),
                INTERIOR: [],
            }
        }
        self._add_creation_info(packed_obj)
        return packed_obj

    @classmethod
    def from_json(cls, data: Dict) -> Rectangle:
        """
        Convert a json dict to Rectangle. Read more about `Supervisely format <https://docs.supervisely.com/data-organization/00_ann_format_navi>`_.

        :param data: Rectangle in json format as a dict.
        :type data: dict
        :return: Rectangle object
        :rtype: :class:`Rectangle<Rectangle>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            figure_json = {
                "points": {
                    "exterior": [
                        [100, 100],
                        [900, 700]
                    ],
                    "interior": []
                }
            }
            figure = sly.Rectangle.from_json(figure_json)
        """
        validation.validate_geometry_points_fields(data)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)

        exterior = data[POINTS][EXTERIOR]
        if len(exterior) != 2:
            raise ValueError(
                '"exterior" field must contain exactly two points to create Rectangle object.'
            )
        [top, bottom] = sorted([exterior[0][1], exterior[1][1]])
        [left, right] = sorted([exterior[0][0], exterior[1][0]])

        return cls(
            top=top,
            left=left,
            bottom=bottom,
            right=right,
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    def crop(self, other: Rectangle) -> List[Rectangle]:
        """
        Crops current Rectangle.

        :param rect: Rectangle object for crop.
        :type rect: Rectangle
        :return: List of Rectangle objects
        :rtype: :class:`List[Rectangle]<Rectangle>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            crop_figures = figure.crop(sly.Rectangle(0, 0, 300, 350))
        """
        top = max(self.top, other.top)
        left = max(self.left, other.left)
        bottom = min(self.bottom, other.bottom)
        right = min(self.right, other.right)
        is_valid = (bottom >= top) and (left <= right)
        return [Rectangle(top=top, left=left, bottom=bottom, right=right)] if is_valid else []

    def _transform(self, transform_fn):
        """ """
        transformed_corners = [transform_fn(p) for p in self.corners]
        rows, cols = zip(*points_to_row_col_list(transformed_corners))
        return Rectangle(
            top=round(min(rows)),
            left=round(min(cols)),
            bottom=round(max(rows)),
            right=round(max(cols)),
        )

    @property
    def corners(
        self,
    ) -> List[PointLocation, PointLocation, PointLocation, PointLocation]:
        """
        Get list of Rectangle corners.

        :return: List of PointLocation objects
        :rtype: :class:`List[PointLocation]<supervisely.geometry.point_location.PointLocation>`

        :Usage Example:

         .. code-block:: python

            corners = figure.corners
            for corner in corners:
                print(corner.row, corner.col)
            # Output:
            # 100 100
            # 100 900
            # 700 900
            # 700 100
        """
        return [
            PointLocation(row=self.top, col=self.left),
            PointLocation(row=self.top, col=self.right),
            PointLocation(row=self.bottom, col=self.right),
            PointLocation(row=self.bottom, col=self.left),
        ]

    def rotate(self, rotator: sly.geometry.image_rotator.ImageRotator) -> Rectangle:
        """
        Rotates current Rectangle.

        :param rotator: ImageRotator object for rotation.
        :type rotator: ImageRotator
        :return: Rectangle object
        :rtype: :class:`Rectangle<Rectangle>`

        :Usage Example:

         .. code-block:: python

            from supervisely.geometry.image_rotator import ImageRotator

            # Remember that Rectangle class object is immutable, and we need to assign new instance of Rectangle to a new variable
            height, width = 300, 400
            rotator = ImageRotator((height, width), 25)
            rotate_figure = figure.rotate(rotator)
        """
        return self._transform(lambda p: rotator.transform_point(p))

    def resize(self, in_size: Tuple[int, int], out_size: Tuple[int, int]) -> Rectangle:
        """
        Resizes current Rectangle.

        :param in_size: Input image size (height, width) to which belongs Rectangle.
        :type in_size: Tuple[int, int]
        :param out_size: Desired output image size (height, width) to which belongs Rectangle.
        :type out_size: Tuple[int, int]
        :return: Rectangle object
        :rtype: :class:`Rectangle<Rectangle>`

        :Usage Example:

         .. code-block:: python

            # Remember that Rectangle class object is immutable, and we need to assign new instance of Rectangle to a new variable
            in_height, in_width = 300, 400
            out_height, out_width = 600, 800
            resize_figure = figure.resize((in_height, in_width), (out_height, out_width))
        """
        return self._transform(lambda p: p.resize(in_size, out_size))

    def scale(self, factor: float) -> Rectangle:
        """
        Scales current Rectangle.

        :param factor: Scale parameter.
        :type factor: float
        :return: Rectangle object
        :rtype: :class:`Rectangle<Rectangle>`

        :Usage Example:

         .. code-block:: python

            # Remember that Rectangle class object is immutable, and we need to assign new instance of Rectangle to a new variable
            scale_figure = figure.scale(0.75)
        """
        return self._transform(lambda p: p.scale(factor))

    def translate(self, drow: int, dcol: int) -> Rectangle:
        """
        Translates current Rectangle.

        :param drow: Horizontal shift.
        :type drow: int
        :param dcol: Vertical shift.
        :type dcol: int
        :return: Rectangle object
        :rtype: :class:`Rectangle<Rectangle>`

        :Usage Example:

         .. code-block:: python

            # Remember that Rectangle class object is immutable, and we need to assign new instance of Rectangle to a new variable
            translate_figure = figure.translate(150, 250)
        """
        return self._transform(lambda p: p.translate(drow, dcol))

    def fliplr(self, img_size: Tuple[int, int]) -> Rectangle:
        """
        Flips current Rectangle in horizontal.

        :param img_size: Input image size (height, width) to which belongs Rectangle.
        :type img_size: Tuple[int, int]
        :return: Rectangle object
        :rtype: :class:`Rectangle<Rectangle>`

        :Usage Example:

         .. code-block:: python

            # Remember that Rectangle class object is immutable, and we need to assign new instance of Rectangle to a new variable
            height, width = 300, 400
            fliplr_figure = figure.fliplr((height, width))
        """
        img_width = img_size[1]
        return Rectangle(
            top=self.top,
            left=(img_width - self.right),
            bottom=self.bottom,
            right=(img_width - self.left),
        )

    def flipud(self, img_size: Tuple[int, int]) -> Rectangle:
        """
        Flips current Rectangle in vertical.

        :param img_size: Input image size (height, width) to which belongs Rectangle.
        :type img_size: Tuple[int, int]
        :return: Rectangle object
        :rtype: :class:`Rectangle<Rectangle>`

        :Usage Example:

         .. code-block:: python

            # Remember that Rectangle class object is immutable, and we need to assign new instance of Rectangle to a new variable
            height, width = 300, 400
            flipud_figure = figure.flipud((height, width))
        """
        img_height = img_size[0]
        return Rectangle(
            top=(img_height - self.bottom),
            left=self.left,
            bottom=(img_height - self.top),
            right=self.right,
        )

    def _draw_impl(self, bitmap: np.ndarray, color, thickness=1, config=None):
        """ """
        self._draw_contour_impl(bitmap, color, thickness=cv2.FILLED, config=config)  # due to cv2

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        """ """
        cv2.rectangle(
            bitmap,
            pt1=(self.left, self.top),
            pt2=(self.right, self.bottom),
            color=color,
            thickness=thickness,
        )

    def to_bbox(self) -> Rectangle:
        """
        Makes a copy of Rectangle.

        :return: Rectangle object
        :rtype: :class:`Rectangle<Rectangle>`

        :Usage Example:

         .. code-block:: python

            # Remember that Rectangle class object is immutable, and we need to assign new instance of Rectangle to a new variable
            new_figure = figure.to_bbox()
        """
        return self.clone()

    @property
    def area(self) -> float:
        """
        Rectangle area.

        :return: Area of current Rectangle object
        :rtype: :class:`float`

        :Usage Example:

         .. code-block:: python

            print(figure.area)
            # Output: 7288.0
        """
        return float(self.width * self.height)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Rectangle:
        """
        Create Rectangle with given array shape.

        :param arr: Numpy array.
        :type arr: np.ndarray
        :return: Rectangle object
        :rtype: :class:`Rectangle<Rectangle>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            np_array = np.zeros((300, 400))
            figure_from_np = sly.Rectangle.from_array(np_array)
        """
        return cls(top=0, left=0, bottom=arr.shape[0] - 1, right=arr.shape[1] - 1)

    # TODO re-evaluate whether we need this, looks trivial.
    @classmethod
    def from_size(cls, size: Tuple[int, int]) -> Rectangle:
        """
        Create Rectangle with given size shape.

        :param size: Input size.
        :type size: Tuple[int, int]
        :return: Rectangle object
        :rtype: :class:`Rectangle<Rectangle>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            size = (300, 400)
            figure_from_size = sly.Rectangle.from_size(size)
        """
        return cls(0, 0, size[0] - 1, size[1] - 1)

    @classmethod
    def from_geometries_list(cls, geometries: List[Geometry]) -> Rectangle:
        """
        Create Rectangle from given geometry objects.

        :param geometries: List of geometry type objects: :class:`Bitmap<supervisely.geometry.bitmap.Bitmap>`, :class:`Cuboid<supervisely.geometry.cuboid.Cuboid>`, :class:`Point<supervisely.geometry.point.Point>`, :class:`Polygon<supervisely.geometry.polygon.Polygon>`, :class:`Polyline<supervisely.geometry.polyline.Polyline>`, :class:`Rectangle<Rectangle>`, :class:`Graph<supervisely.geometry.graph.GraphNodes>`.
        :type geometries: List[Geometry]
        :return: Rectangle object
        :rtype: :class:`Rectangle<Rectangle>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            geom_objs = [sly.Point(100, 200), sly.Polyline([sly.PointLocation(730, 2104), sly.PointLocation(2479, 402)])]
            figure_from_geom_objs = sly.Rectangle.from_geometries_list(geom_objs)
        """
        if geometries is None or len(geometries) == 0:
            raise ValueError("No geometries provided to create a Rectangle.")
        bboxes = [g.to_bbox() for g in geometries]
        top = min(bbox.top for bbox in bboxes)
        left = min(bbox.left for bbox in bboxes)
        bottom = max(bbox.bottom for bbox in bboxes)
        right = max(bbox.right for bbox in bboxes)
        return cls(top=top, left=left, bottom=bottom, right=right)

    @property
    def left(self) -> int:
        """
        Minimal horizontal value of Rectangle.

        :return: Minimal horizontal value
        :rtype: :class:`int`

        :Usage Example:

         .. code-block:: python

            print(figure.left)
            # Output: 100
        """
        return self._points[0].col

    @property
    def right(self) -> int:
        """
        Maximal horizontal value of Rectangle.

        :return: Maximal horizontal value
        :rtype: :class:`int`

        :Usage Example:

         .. code-block:: python

            print(figure.right)
            # Output: 900
        """
        return self._points[1].col

    @property
    def top(self) -> int:
        """
        Minimal vertical value of Rectangle.

        :return: Minimal vertical value
        :rtype: :class:`int`

        :Usage Example:

         .. code-block:: python

            print(rectangle.top)
            # Output: 100
        """
        return self._points[0].row

    @property
    def bottom(self) -> int:
        """
        Maximal vertical value of Rectangle.

        :return: Maximal vertical value
        :rtype: :class:`int`

        :Usage Example:

         .. code-block:: python

            print(figure.bottom)
            # Output: 700
        """
        return self._points[1].row

    @property
    def center(self) -> PointLocation:
        """
        Center of Rectangle.

        :return: PointLocation object
        :rtype: :class:`PointLocation<supervisely.geometry.point_location.PointLocation>`

        :Usage Example:

         .. code-block:: python

            center = figure.center()
        """
        return PointLocation(row=(self.top + self.bottom) // 2, col=(self.left + self.right) // 2)

    @property
    def width(self) -> int:
        """
        Width of Rectangle.

        :return: Width
        :rtype: :class:`int`

        :Usage Example:

         .. code-block:: python

            print(figure.width)
            # Output: 801
        """
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        """
        Height of Rectangle

        :return: Height
        :rtype: :class:`int`

        :Usage Example:

         .. code-block:: python

            print(figure.height)
            # Output: 601
        """
        return self.bottom - self.top + 1

    def contains(self, rect: Rectangle) -> bool:
        """
        Checks if Rectangle contains a given Rectangle object.

        :param rect: Rectangle object.
        :type rect: Rectangle
        :return: True if Rectangle contains given Rectangle object, otherwise False
        :rtype: :class:`bool`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            rect = sly.Rectangle(200, 250, 400, 500))
            print(figure.contains(rect))
            # Output: True
        """
        return (
            self.left <= rect.left
            and self.right >= rect.right
            and self.top <= rect.top
            and self.bottom >= rect.bottom
        )

    def contains_point_location(self, pt: PointLocation) -> bool:
        """
        Checks if Rectangle contains a given PointLocation object.

        :param pt: PointLocation object.
        :type pt: PointLocation
        :return: True if Rectangle contains given PointLocation object, otherwise False
        :rtype: :class:`bool`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            pt = sly.PointLocation(250, 300))
            print(figure.contains_point_location(pt))
            # Output: True
        """
        return (self.left <= pt.col <= self.right) and (self.top <= pt.row <= self.bottom)

    def to_size(self) -> Tuple[int, int]:
        """
        Height and width of Rectangle.

        :return: Height and width of Rectangle object
        :rtype: :class:`Tuple[int, int]`

        :Usage Example:

         .. code-block:: python

            height, width = figure.to_size()
            print(height, width)
            # Output: 700 900
        """
        return self.height, self.width

    def get_cropped_numpy_slice(self, data: np.ndarray) -> np.ndarray:
        """
        Slice of given numpy array with Rectangle.

        :param data: Numpy array.
        :type data: np.ndarray
        :return: Sliced numpy array
        :rtype: :class:`np.ndarray<np.ndarray>`

        :Usage Example:

         .. code-block:: python

            np_slice = np.zeros((200, 500))
            mask_slice = figure.get_cropped_numpy_slice(np_slice)
            print(mask_slice.shape)
            # Output: (199, 499)
        """
        return data[self.top : (self.bottom + 1), self.left : (self.right + 1), ...]

    def intersects_with(self, rect: Rectangle) -> bool:
        """
        Checks intersects Rectangle with given Rectangle object or not.

        :param rect: Rectangle object.
        :type rect: Rectangle
        :return: True if given Rectangle object intersects with Rectangle, otherwise False
        :rtype: :class:`bool`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            rect = sly.Rectangle(90, 90, 400, 500)
            print(figure.intersects_with(rect))
            # Output: True
        """
        if self.left > rect.right or self.right < rect.left:
            return False
        if self.top > rect.bottom or self.bottom < rect.top:
            return False
        return True

    @classmethod
    def allowed_transforms(cls):
        """ """
        from supervisely.geometry.alpha_mask import AlphaMask
        from supervisely.geometry.any_geometry import AnyGeometry
        from supervisely.geometry.bitmap import Bitmap
        from supervisely.geometry.polygon import Polygon

        return [AlphaMask, AnyGeometry, Bitmap, Polygon]

    @classmethod
    def _round_subpixel_coordinates(
        cls,
        top: Union[int, float],
        left: Union[int, float],
        bottom: Union[int, float],
        right: Union[int, float],
    ) -> Tuple[int, int, int, int]:
        """
        Apply rounding logic to the Rectangle coordinates. Coordinates will remain in subpixel precision.

        Top will be rounded down if the decimal part is lesser than 0.7, it will include the vertical pixel if it is in the range, otherwise it will be rounded up and this pixel will not be included.
        Left will be rounded down if the decimal part is lesser than 0.7, it will include the horizontal pixel if it is in the range, otherwise it will be rounded up and this pixel will not be included.
        Bottom will be rounded up if the decimal part is greater than 0.3, it will include the vertical pixel if it is in the range, otherwise it will be rounded down and this pixel will not be included.
        Right will be rounded up if the decimal part is greater than 0.3, it will include the horizontal pixel if it is in the range, otherwise it will be rounded down and this pixel will not be included.

        Example:
        Input coordinates:
        - top = 1.55, left = 1.74, bottom = 4.63, right = 3.76
        Output coordinates:
        - top = 1, left = 2, bottom = 5, right = 4
        - top will be rounded down to 2, left will be rounded down to 2, bottom will be rounded up to 6, right will be rounded down to 6


        Subpixel coordinate system:
            0   1   2   3   4   5
        0   +---+---+---+---+---+
            |   |   |   |   |   |
        1   +---+---+---+---+---+
            |   |   | x | x |   |
        2   +---+---+---+---+---+
            |   |   | x | x |   |
        3   +---+---+---+---+---+
            |   |   | x | x |   |
        4   +---+---+---+---+---+
            |   |   | x | x |   |
        5   +---+---+---+---+---+
                      x   x

        :param top: Minimal vertical value of Rectangle object.
        :type top: Union[int, float]
        :param left: Minimal horizontal value of Rectangle object.
        :type left: Union[int, float]
        :param bottom: Maximal vertical value of Rectangle object.
        :type bottom: Union[int, float]
        :param right: Maximal vertical value of Rectangle object.
        :type right: Union[int, float]
        :return: Rounded rectangle coordinates
        :rtype: Tuple[int, int, int, int]
        """
        RIGHT_OVERLAP = 0.3
        LEFT_OVERLAP = 1 - RIGHT_OVERLAP

        # Check case if all coords in 1 pixel range
        if int(top) == int(bottom) and int(left) == int(right):
            return int(top), int(left), int(bottom), int(right)

        if top % 1 >= LEFT_OVERLAP:
            top = ceil(top)
        else:
            top = floor(top)

        if left % 1 >= LEFT_OVERLAP:
            left = ceil(left)
        else:
            left = floor(left)

        if bottom % 1 >= RIGHT_OVERLAP:
            bottom = ceil(bottom)
        else:
            bottom = floor(bottom)
        if right % 1 >= RIGHT_OVERLAP:
            right = ceil(right)
        else:
            right = floor(right)
        return top, left, bottom, right

    @classmethod
    def _to_pixel_coordinate_system_json(cls, data: Dict, image_size: List[int]) -> Dict:
        """
        Convert Rectangle from subpixel precision to pixel precision by subtracting a subpixel offset from the coordinates.

        Points order in json format: [[left, top], [right, bottom]]

        In the labeling tool, labels are created with subpixel precision,
        which means that the coordinates of the rectangle corners (top, left and bottom, right) can have decimal values representing fractions of a pixel.
        However, in Supervisely SDK, geometry coordinates are represented using pixel precision, where the coordinates are integers representing whole pixels.

        Example:
        Step 1. Input coordinates:
        - top = 1.55, left = 1.74, bottom = 4.63, right = 3.76

        Step 2. Round the coordinates (still remain in subpixel precision):
        - top = 1, left = 2, bottom = 5, right = 4
        - top will be rounded down to 2, left will be rounded down to 2, bottom will be rounded up to 6, right will be rounded down to 6

        Draw coordinates in pixel coordinate system:
            0   1   2   3   4   5
        0   +---+---+---+---+---+
            |   |   |   |   |   |
        1   +---+---+---+---+---+
            |   |   | x | x |   |
        2   +---+---+---+---+---+
            |   |   | x | x |   |
        3   +---+---+---+---+---+
            |   |   | x | x |   |
        4   +---+---+---+---+---+
            |   |   | x | x |   |
        5   +---+---+---+---+---+
                      x   x

        Step 3. Convert to pixel coordinates by subtracting a subpixel offset:
        - top = 1, left = 2, bottom = 4, right = 3

        Draw coordinates in pixel coordinate system:
            0   1   2   3   4   5
        0   +---+---+---+---+---+
            |   |   |   |   |   |
        1   +---+---+---+---+---+
            |   |   | x | x |   |
        2   +---+---+---+---+---+
            |   |   | x | x |   |
        3   +---+---+---+---+---+
            |   |   | x | x |   |
        4   +---+---+---+---+---+
            |   |   | x | x |   |
        5   +---+---+---+---+---+

        :param data: Json data with geometry config.
        :type data: :class:`dict`
        :param image_size: Image size in pixels (height, width).
        :type image_size: List[int]
        :return: Json data with coordinates converted to pixel coordinate system.
        :rtype: :class:`dict`
        """
        data = deepcopy(data)  # Avoid modifying the original data
        height, width = image_size[:2]

        exterior = data[POINTS][EXTERIOR]
        [top, bottom] = sorted([exterior[0][1], exterior[1][1]])
        [left, right] = sorted([exterior[0][0], exterior[1][0]])

        top, left, bottom, right = cls._round_subpixel_coordinates(top, left, bottom, right)
        right = max(left, right - 1)
        bottom = max(top, bottom - 1)
        data[POINTS][EXTERIOR] = [[left, top], [right, bottom]]
        return data

    @classmethod
    def _to_subpixel_coordinate_system_json(cls, data: Dict) -> Dict:
        """
        Convert Rectangle from pixel precision to subpixel precision by adding a subpixel offset to the coordinates.

        Points order in json format: [[left, top], [right, bottom]]

        In the labeling tool, labels are created with subpixel precision,
        which means that the coordinates of the rectangle corners (top, left and bottom, right) can have decimal values representing fractions of a pixel.
        However, in Supervisely SDK, geometry coordinates are represented using pixel precision, where the coordinates are integers representing whole pixels.

        :param data: Json data with geometry config.
        :type data: :class:`dict`
        :return: Json data with coordinates converted to subpixel coordinate system.
        :rtype: :class:`dict`
        """
        data = deepcopy(data)  # Avoid modifying the original data

        exterior = data[POINTS][EXTERIOR]
        [top, bottom] = sorted([exterior[0][1], exterior[1][1]])
        [left, right] = sorted([exterior[0][0], exterior[1][0]])

        right = max(left, right + 1)
        bottom = max(top, bottom + 1)
        data[POINTS][EXTERIOR] = [[left, top], [right, bottom]]
        return data

    # def _to_pixel_coordinate_system(self) -> Rectangle:
    #     """
    #     Convert Rectangle from subpixel precision to pixel precision by subtracting a subpixel offset from the coordinates.

    #     In the labeling tool, labels are created with subpixel precision,
    #     which means that the coordinates of the rectangle corners (top, left and bottom, right) can have decimal values representing fractions of a pixel.
    #     However, in Supervisely SDK, geometry coordinates are represented using pixel precision, where the coordinates are integers representing whole pixels.

    #     Example:
    #     Step 1. Input coordinates:
    #     - top = 1.55, left = 1.74, bottom = 4.63, right = 3.76

    #     Step 2. Round the coordinates:
    #     - top = 1, left = 2, bottom = 5, right = 4
    #     - top will be rounded down to 2, left will be rounded down to 2, bottom will be rounded up to 6, right will be rounded down to 6

    #     Draw coordinates in pixel coordinate system:
    #         0   1   2   3   4   5
    #     0   +---+---+---+---+---+
    #         |   |   |   |   |   |
    #     1   +---+---+---+---+---+
    #         |   |   | x | x |   |
    #     2   +---+---+---+---+---+
    #         |   |   | x | x |   |
    #     3   +---+---+---+---+---+
    #         |   |   | x | x |   |
    #     4   +---+---+---+---+---+
    #         |   |   | x | x |   |
    #     5   +---+---+---+---+---+
    #                   x   x

    #     Step 3. Convert to pixel coordinates by subtracting a subpixel offset:
    #     - top = 1, left = 2, bottom = 4, right = 3

    #     Draw coordinates in pixel coordinate system:
    #         0   1   2   3   4   5
    #     0   +---+---+---+---+---+
    #         |   |   |   |   |   |
    #     1   +---+---+---+---+---+
    #         |   |   | x | x |   |
    #     2   +---+---+---+---+---+
    #         |   |   | x | x |   |
    #     3   +---+---+---+---+---+
    #         |   |   | x | x |   |
    #     4   +---+---+---+---+---+
    #         |   |   | x | x |   |
    #     5   +---+---+---+---+---+

    #     :return: New instance of Rectangle object with corners in pixel format.
    #     :rtype: :class:`Rectangle<Rectangle>`
    #     """
    #     left = floor(self.left)
    #     top = floor(self.top)
    #     right = max(left, floor(self.right) - 1)
    #     bottom = max(top, floor(self.bottom) - 1)
    #     return Rectangle(
    #         top=top,
    #         left=left,
    #         bottom=bottom,
    #         right=right,
    #         sly_id=self.sly_id,
    #         class_id=self.class_id,
    #         labeler_login=self.labeler_login,
    #         updated_at=self.updated_at,
    #         created_at=self.created_at,
    #     )

    # def _to_subpixel_coordinate_system(self) -> Rectangle:
    #     """
    #     Convert Rectangle from pixel precision to subpixel precision by adding a subpixel offset to the coordinates.

    #     In the labeling tool, labels are created with subpixel precision,
    #     which means that the coordinates of the rectangle corners (top, left and bottom, right) can have decimal values representing fractions of a pixel.
    #     However, in Supervisely SDK, geometry coordinates are represented using pixel precision, where the coordinates are integers representing whole pixels.

    #     :return: New instance of Rectangle object with corners in subpixel format.
    #     :rtype: :class:`Rectangle<Rectangle>`
    #     """
    #     left = self.left
    #     top = self.top
    #     right = self.right + 1
    #     bottom = self.bottom + 1

    #     return Rectangle(
    #         top=top,
    #         left=left,
    #         bottom=bottom,
    #         right=right,
    #         sly_id=self.sly_id,
    #         class_id=self.class_id,
    #         labeler_login=self.labeler_login,
    #         updated_at=self.updated_at,
    #         created_at=self.created_at,
    #     )
