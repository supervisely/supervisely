# coding: utf-8

# docs
from __future__ import annotations

from copy import deepcopy
from math import ceil, floor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from supervisely.geometry import validation
from supervisely.geometry.constants import (
    ANGLE,
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
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle


class OrientedBBox(Rectangle):
    """
    OrientedBBox geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`OrientedBBox<OrientedBBox>` class object is immutable.

    :param top: Minimal vertical value of OrientedBBox object.
    :type top: int or float
    :param left: Minimal horizontal value of OrientedBBox object.
    :type left: int or float
    :param bottom: Maximal vertical value of OrientedBBox object.
    :type bottom: int or float
    :param right: Maximal vertical value of OrientedBBox object.
    :type right: int or float
    :param angle: Angle of rotation in degrees. Positive values mean clockwise rotation.
    :type angle: int or float, optional
    :param sly_id: OrientedBBox ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which OrientedBBox belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created OrientedBBox.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when OrientedBBox was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when OrientedBBox was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :raises: :class:`ValueError`. OrientedBBox top argument must have less or equal value then bottom, left argument must have less or equal value then right

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        top = 100
        left = 100
        bottom = 700
        right = 900
        angle = 15
        figure = sly.OrientedBBox(top, left, bottom, right, angle=angle)
    """

    @staticmethod
    def geometry_name():
        """ """
        return "oriented_bbox"

    def __init__(
        self,
        top: int,
        left: int,
        bottom: int,
        right: int,
        angle: Union[int, float] = 0,
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[int] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):

        super().__init__(
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
        self._angle = angle

    @property
    def angle(self) -> Union[int, float]:
        """
        Angle of rotation in degrees. Positive values mean clockwise rotation.

        :return: Angle of rotation
        :rtype: int or float
        :Usage example:

         .. code-block:: python

            angle = oriented_bbox.angle
        """
        return self._angle

    def to_json(self) -> Dict:
        """
        Convert the OrientedBBox to a json dict. Read more about `Supervisely format <https://docs.supervisely.com/data-organization/00_ann_format_navi>`_.

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
            #    },
            #    "angle": 15
            # }
        """
        packed_obj = {
            POINTS: {
                EXTERIOR: points_to_row_col_list(self._points, flip_row_col_order=True),
                INTERIOR: [],
            },
            ANGLE: self._angle,
        }
        self._add_creation_info(packed_obj)
        return packed_obj

    @classmethod
    def from_json(cls, data: Dict) -> OrientedBBox:
        """
        Convert a json dict to OrientedBBox. Read more about `Supervisely format <https://docs.supervisely.com/data-organization/00_ann_format_navi>`_.

        :param data: OrientedBBox in json format as a dict.
        :type data: dict
        :return: OrientedBBox object
        :rtype: :class:`OrientedBBox<OrientedBBox>`
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
                },
                "angle": 15
            }
            figure = sly.OrientedBBox.from_json(figure_json)
        """
        validation.validate_geometry_points_fields(data)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        angle = data.get(ANGLE, 0)

        exterior = data[POINTS][EXTERIOR]
        if len(exterior) != 2:
            raise ValueError(
                '"exterior" field must contain exactly two points to create OrientedBBox object.'
            )
        [top, bottom] = sorted([exterior[0][1], exterior[1][1]])
        [left, right] = sorted([exterior[0][0], exterior[1][0]])

        return cls(
            top=top,
            left=left,
            bottom=bottom,
            right=right,
            angle=angle,
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    def to_bbox(self) -> Rectangle:
        """
        Convert the OrientedBBox to the axis-aligned :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>` that fully contains the OrientedBBox.

        :return: Axis-aligned Rectangle object
        :rtype: :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>`
        :Usage example:
            .. code-block:: python
    
                axis_aligned_bbox = oriented_bbox.to_bbox()
            """
        if self._angle % 360 == 0:
            return Rectangle(
                top=self._top,
                left=self._left,
                bottom=self._bottom,
                right=self._right,
                sly_id=self._id,
                class_id=self._class_id,
                labeler_login=self._labeler_login,
                updated_at=self._updated_at,
                created_at=self._created_at,
            )

        angle_rad = np.deg2rad(self._angle)

        cos_angle = abs(np.cos(angle_rad))
        sin_angle = abs(np.sin(angle_rad))

        new_w = self.width * cos_angle + self.height * sin_angle
        new_h = self.width * sin_angle + self.height * cos_angle

        new_left = self.center.col - new_w / 2.0
        new_right = self.center.col + new_w / 2.0
        new_top = self.center.row - new_h / 2.0
        new_bottom = self.center.row + new_h / 2.0

        return Rectangle(
            top=new_top,
            left=new_left,
            bottom=new_bottom,
            right=new_right,
            sly_id=self._id,
            class_id=self._class_id,
            labeler_login=self._labeler_login,
            updated_at=self._updated_at,
            created_at=self._created_at,
        )

    def contains_point_location(self, point: PointLocation) -> bool:
        """
        Check if the OrientedBBox contains the given point.

        :param point: PointLocation object
        :type point: :class:`PointLocation<supervisely.geometry.point_location.PointLocation>`
        :return: True if the point is inside the OrientedBBox, False otherwise
        :rtype: bool
        :Usage example:

         .. code-block:: python

            point = sly.PointLocation(150, 200)
            is_inside = oriented_bbox.contains_point_location(point)
        """
        # Rotate point in the opposite direction around the center of the oriented bbox
        angle_rad = np.deg2rad(-self._angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # Translate point to origin
        translated_x = point.col - self.center.col
        translated_y = point.row - self.center.row

        # Rotate point
        rotated_x = translated_x * cos_angle - translated_y * sin_angle
        rotated_y = translated_x * sin_angle + translated_y * cos_angle

        # Translate point back
        final_x = rotated_x + self.center.col
        final_y = rotated_y + self.center.row

        # Check if the rotated point is within the axis-aligned bbox
        return (
            self._left <= final_x <= self._right and self._top <= final_y <= self._bottom
        )

    def contains_point(self, geometry: Geometry) -> bool:
        """
        Check if the OrientedBBox contains the given point.

        :param geometry: PointLocation object
        :type geometry: :class:`Geometry<supervisely.geometry.geometry.Geometry>`
        :return: True if the point is inside the OrientedBBox, False otherwise
        :rtype: bool
        :Usage example:

         .. code-block:: python

            point = sly.PointLocation(150, 200)
            is_inside = oriented_bbox.contains_point(point)
        """
        if isinstance(geometry, PointLocation):
            return self.contains_point_location(geometry)
        elif isinstance(geometry, Rectangle):
            return self.contains_rectangle(geometry)
        elif isinstance(geometry, OrientedBBox):
            return self.contains_obb(geometry)
        else:
            raise TypeError(
                "Unsupported geometry type for contains_point method. "
                "Supported types are PointLocation, Rectangle, and OrientedBBox."
            )

    def contains_obb(self, obb: OrientedBBox) -> bool:
        """
        Check if the OrientedBBox contains the given OrientedBBox.

        :param obb: OrientedBBox object
        :type obb: :class:`OrientedBBox<supervisely.geometry.oriented_bbox.OrientedBBox>`
        :return: True if the OrientedBBox is inside the OrientedBBox, False otherwise
        :rtype: bool
        :Usage example:

         .. code-block:: python

            obb2 = sly.OrientedBBox(150, 200, 400, 500, angle=10)
            is_inside = obb1.contains_obb(obb2)
        """
        # Get the corners of the obb
        corners = obb.calculate_rotated_corners()

        # Check if all corners are inside the current obb
        for corner in corners:
            if not self.contains_point_location(corner):
                return False
        return True

    @staticmethod
    def _calculate_rotated_corners(obb: OrientedBBox) -> List[PointLocation]:
        """
        Get the corners of the OrientedBBox.

        :return: List of corners as (x, y) tuples
        :rtype: List[Tuple[float, float]]
        :Usage example:

         .. code-block:: python

            corners = oriented_bbox.calculate_rotated_corners()
        """
        angle_rad = np.deg2rad(obb._angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        rotated_corners = []
        for corner in obb.corners:  # [Top-left, Top-right, Bottom-right, Bottom-left]
            rotated_x = corner.col * cos_angle - corner.row * sin_angle
            rotated_y = corner.col * sin_angle + corner.row * cos_angle

            final_x = rotated_x + obb.center.col
            final_y = rotated_y + obb.center.row

            rotated_corners.append(PointLocation(row=final_y, col=final_x))

        return rotated_corners

    def calculate_rotated_corners(self) -> List[PointLocation]:
        """
        Get the corners of the OrientedBBox.

        :return: List of corners as PointLocation objects
        :rtype: List[:class:`PointLocation<supervisely.geometry.point_location.PointLocation>`]
        :Usage example:

         .. code-block:: python

            corners = oriented_bbox.calculate_rotated_corners()
        """
        return self._calculate_rotated_corners(self)

    def contains_rectangle(self, rectangle: Rectangle) -> bool:
        """
        Check if the OrientedBBox contains the given Rectangle.

        :param rectangle: Rectangle object
        :type rectangle: :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>`
        :return: True if the Rectangle is inside the OrientedBBox, False otherwise
        :rtype: bool
        :Usage example:

         .. code-block:: python

            rectangle = sly.Rectangle(150, 200, 400, 500)
            is_inside = obb.contains_rectangle(rectangle)
        """
        # Get the corners of the rectangle
        corners = rectangle.corners  # [Top-left, Top-right, Bottom-right, Bottom-left]

        # Check if all corners are inside the current obb
        for corner in corners:
            if not self.contains_point_location(corner):
                return False
        return True

    def from_bbox(cls, bbox: Rectangle) -> OrientedBBox:
        """
        Create OrientedBBox from given Rectangle.

        :param bbox: Rectangle object.
        :type bbox: :class:`Rectangle<supervisely.geometry.rectangle.Rectangle>`
        :return: OrientedBBox object
        :rtype: :class:`OrientedBBox<OrientedBBox>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            axis_aligned_bbox = sly.Rectangle(100, 100, 700, 900)
            figure_from_bbox = sly.OrientedBBox.from_bbox(axis_aligned_bbox)
        """
        return cls(
            top=bbox.top,
            left=bbox.left,
            bottom=bbox.bottom,
            right=bbox.right,
            angle=0,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray, angle: Union[int, float] = 0) -> OrientedBBox:
        """
        Create OrientedBBox with given array shape.

        :param arr: Numpy array.
        :type arr: np.ndarray
        :return: OrientedBBox object
        :rtype: :class:`OrientedBBox<OrientedBBox>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            np_array = np.zeros((300, 400))
            figure_from_np = sly.OrientedBBox.from_array(np_array, angle=15)
        """
        return cls(top=0, left=0, bottom=arr.shape[0] - 1, right=arr.shape[1] - 1, angle=angle)

    def get_cropped_numpy_slice(self, data: np.ndarray) -> np.ndarray:
        """
        Slice of given numpy array with OrientedBBox align bbox.

        :param data: Numpy array.
        :type data: np.ndarray
        :return: Sliced numpy array
        :rtype: :class:`np.ndarray<np.ndarray>`

        :Usage Example:

         .. code-block:: python

            np_slice = np.zeros((200, 500))
            mask_slice = figure.get_cropped_numpy_slice(np_slice)
            print(mask_slice.shape)
        """
        axis_aligned_bbox = self.to_bbox()
        top = max(0, floor(axis_aligned_bbox.top))
        left = max(0, floor(axis_aligned_bbox.left))
        bottom = min(data.shape[0], ceil(axis_aligned_bbox.bottom))
        right = min(data.shape[1], ceil(axis_aligned_bbox.right))
        return data[top:bottom, left:right]

    @classmethod
    def allowed_transforms(cls):
        """ """
        from supervisely.geometry.alpha_mask import AlphaMask
        from supervisely.geometry.any_geometry import AnyGeometry
        from supervisely.geometry.bitmap import Bitmap

        return [AlphaMask, AnyGeometry, Bitmap, Polygon, OrientedBBox]

    def crop(self, clip: OrientedBBox | Rectangle, return_type: Polygon | Rectangle = Polygon) -> List[Polygon | Rectangle]:
        """Crop the OrientedBBox by another OrientedBBox using the Sutherland-Hodgman algorithm."""
        subject_corners = self._calculate_rotated_corners(self)
        if isinstance(clip, Rectangle):
            clip_corners = clip.corners
        else:
            clip_corners = self._calculate_rotated_corners(clip)

        def inside(p: PointLocation, edge_start: PointLocation, edge_end: PointLocation) -> bool:
            cx1, cy1 = edge_start.col, edge_start.row
            cx2, cy2 = edge_end.col, edge_end.row
            px, py = p.col, p.row
            return (cx2 - cx1) * (py - cy1) > (cy2 - cy1) * (px - cx1)

        def compute_intersection(
            p1: PointLocation,
            p2: PointLocation,
            edge_start: PointLocation,
            edge_end: PointLocation,
        ) -> Optional[PointLocation]:
            x1, y1 = p1.col, p1.row
            x2, y2 = p2.col, p2.row
            x3, y3 = edge_start.col, edge_start.row
            x4, y4 = edge_end.col, edge_end.row
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10: return None
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            return PointLocation(row=y1 + t * (y2 - y1), col=x1 + t * (x2 - x1))

        output = subject_corners[:]
        n = len(clip_corners)

        for i in range(n):
            edge_start = clip_corners[i]
            edge_end = clip_corners[(i + 1) % n]
            input_list = output
            output = []

            m = len(input_list)
            for j in range(m):
                curr = input_list[j]
                prev = input_list[(j - 1) % m]

                curr_inside = inside(curr, edge_start, edge_end)
                prev_inside = inside(prev, edge_start, edge_end)

                if curr_inside:
                    if not prev_inside:
                        inter = compute_intersection(prev, curr, edge_start, edge_end)
                        if inter: output.append(inter)
                    output.append(curr)
                elif prev_inside:
                    inter = compute_intersection(prev, curr, edge_start, edge_end)
                    if inter: output.append(inter)

            if len(output) < 3: return []

        polygon = Polygon(output, [])
        if return_type == Rectangle:
            return [polygon.to_bbox()]
        return [polygon]
