# coding: utf-8

import cv2
import numpy as np

from supervisely_lib.geometry.constants import FACES, POINTS
from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.point_location import points_to_row_col_list, row_col_list_to_points
from supervisely_lib.geometry.rectangle import Rectangle


class CuboidFace:
    def __init__(self, a, b, c, d):
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def to_json(self):
        return [self.a, self.b, self.c, self.d]

    @classmethod
    def from_json(cls, data):
        if len(data) != 4:
            raise ValueError(f'CuboidFace JSON data must have exactly 4 indices, instead got {len(data)!r}.')
        return cls(data[0], data[1], data[2], data[3])

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def d(self):
        return self._d

    def tolist(self):
        return [self.a, self.b, self.c, self.d]


class Cuboid(Geometry):
    @staticmethod
    def geometry_name():
        return 'cuboid'

    def __init__(self, points, faces):
        """
        Args:
            points: iterable of PointLocation objects.
            faces: iterable of CuboidFace objects.
        """

        points = list(points)
        faces = list(faces)

        if len(faces) != 3:
            raise ValueError(f'A cuboid must have exactly 3 faces. Instead got {len(faces)} faces.')

        for face in faces:
            for point_idx in (face.a, face.b, face.c, face.d):
                if point_idx >= len(points) or point_idx < 0:
                    raise ValueError(f'Point index is out of bounds for cuboid face. Got {len(points)!r} points, but '
                                     f'the index is {point_idx!r}.')

        self._points = points
        self._faces = faces

    """
    Implementation of all methods from Geometry
    """

    @property
    def points(self):
        return self._points.copy()

    @property
    def faces(self):
        return self._faces.copy()

    def to_json(self):
        packed_obj = {
            POINTS: points_to_row_col_list(self._points, flip_row_col_order=True),
            FACES: [face.to_json() for face in self._faces]
        }
        return packed_obj

    @classmethod
    def from_json(cls, data):
        for k in [POINTS, FACES]:
            if k not in data:
                raise ValueError(f'Field {k!r} not found in Cuboid JSON data.')

        points = row_col_list_to_points(data[POINTS], flip_row_col_order=True)
        faces = [CuboidFace.from_json(face_json) for face_json in data[FACES]]
        return cls(points=points, faces=faces)

    def crop(self, rect):
        is_all_nodes_inside = all(
            rect.contains_point_location(self._points[p]) for face in self._faces for p in face.tolist())
        return [self] if is_all_nodes_inside else []

    def _transform(self, transform_fn):
        return Cuboid(points=[transform_fn(p) for p in self.points], faces=self.faces)

    def rotate(self, rotator):
        return self._transform(lambda p: rotator.transform_point(p))

    def resize(self, in_size, out_size):
        return self._transform(lambda p: p.resize(in_size, out_size))

    def scale(self, factor):
        return self._transform(lambda p: p.scale(factor))

    def translate(self, drow, dcol):
        return self._transform(lambda p: p.translate(drow, dcol))

    def fliplr(self, img_size):
        return self._transform(lambda p: p.fliplr(img_size))

    def flipud(self, img_size):
        return self._transform(lambda p: p.flipud(img_size))

    def _draw_impl(self, bitmap: np.ndarray, color, thickness=1, config=None):
        bmp_to_draw = np.zeros(bitmap.shape[:2], np.uint8)
        for contour in self._contours_list():
            cv2.fillPoly(bmp_to_draw, pts=[np.array(contour, dtype=np.int32)], color=1)
        bool_mask = bmp_to_draw.astype(bool)
        bitmap[bool_mask] = color

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        contours_np_list = [np.array(contour, dtype=np.int32) for contour in self._contours_list()]
        cv2.polylines(bitmap, pts=contours_np_list, isClosed=True, color=color, thickness=thickness)

    def _contours_list(self):
        return [points_to_row_col_list([self._points[idx] for idx in face.tolist()], flip_row_col_order=True)
                for face in self._faces]

    def to_bbox(self):
        points_np = np.array([[self._points[p].row, self._points[p].col]
                              for face in self._faces for p in face.tolist()])
        rows, cols = points_np[:, 0], points_np[:, 1]
        return Rectangle(top=round(min(rows).item()), left=round(min(cols).item()), bottom=round(max(rows).item()),
                         right=round(max(cols).item()))

    @property
    def area(self):
        bbox = self.to_bbox()
        canvas = np.zeros([bbox.bottom + 1, bbox.right + 1], dtype=np.bool)
        self.draw(canvas, True)
        return float(np.sum(canvas))
