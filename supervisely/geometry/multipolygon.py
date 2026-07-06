# coding: utf-8
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

from supervisely.geometry.constants import (
    CLASS_ID,
    CREATED_AT,
    EXTERIOR,
    ID,
    INTERIOR,
    LABELER_LOGIN,
    PARTS,
    UPDATED_AT,
)
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.point_location import (
    PointLocation,
    points_to_row_col_list,
    row_col_list_to_points,
)
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle


PolygonLike = Union[
    Polygon,
    Dict,
    Tuple[
        Union[List[PointLocation], List[List[int]], List[Tuple[int, int]]],
        Union[List[List[PointLocation]], List[List[List[int]]], List[List[Tuple[int, int]]]],
    ],
]


class Multipolygon(Geometry):
    """A 2D geometry that groups several polygon parts into one label. Immutable."""

    @staticmethod
    def geometry_name():
        """
        Returns the name of the geometry.

        :returns: name of the geometry
        :rtype: str
        """
        return "multipolygon"

    def __init__(
        self,
        parts: List[PolygonLike],
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        """
        Multipolygon geometry for a single :class:`~supervisely.annotation.label.Label`.

        :param parts: Polygon parts. Each part can be a :class:`~supervisely.geometry.polygon.Polygon`,
            a dict with ``exterior`` and optional ``interior`` fields, or a tuple ``(exterior, interior)``.
        :type parts: List[Polygon], List[dict], List[tuple]
        :param sly_id: Multipolygon ID in Supervisely server.
        :type sly_id: int, optional
        :param class_id: ID of ObjClass to which Multipolygon belongs.
        :type class_id: int, optional
        :param labeler_login: Login of the user who created Multipolygon.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when Multipolygon was modified last.
        :type updated_at: str, optional
        :param created_at: Date and Time when Multipolygon was created.
        :type created_at: str, optional
        :raises ValueError: if no parts are provided
        """
        if not isinstance(parts, list):
            raise TypeError('Argument "parts" must be a list of Polygon objects or polygon data')
        if len(parts) == 0:
            raise ValueError('Argument "parts" must contain at least one polygon')

        self._parts = [self._make_polygon(part) for part in parts]
        super().__init__(
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    @staticmethod
    def _make_polygon(part: PolygonLike) -> Polygon:
        if isinstance(part, Polygon):
            return part.clone()
        if isinstance(part, dict):
            return Polygon(part[EXTERIOR], part.get(INTERIOR, []))
        if isinstance(part, tuple) and len(part) == 2:
            exterior, interior = part
            return Polygon(exterior, interior)
        raise TypeError(
            'Multipolygon parts must be Polygon objects, dicts with "exterior"/"interior", '
            'or tuples with "(exterior, interior)"'
        )

    @property
    def parts(self) -> List[Polygon]:
        """
        Polygon parts of the Multipolygon.

        :returns: cloned list of Polygon parts
        :rtype: List[:class:`~supervisely.geometry.polygon.Polygon`]
        """
        return [part.clone() for part in self._parts]

    def to_polygons(self) -> List[Polygon]:
        """
        Converts Multipolygon into several Polygon geometries.

        :returns: cloned list of Polygon parts
        :rtype: List[:class:`~supervisely.geometry.polygon.Polygon`]
        """
        return self.parts

    def to_json(self) -> Dict:
        """
        Convert the Multipolygon to a json dict in Supervisely format.

        :returns: Json format as a dict
        :rtype: dict
        """
        packed_obj = {
            PARTS: [
                {
                    EXTERIOR: points_to_row_col_list(part.exterior, flip_row_col_order=True),
                    INTERIOR: [
                        points_to_row_col_list(i, flip_row_col_order=True)
                        for i in part.interior
                    ],
                }
                for part in self._parts
            ],
        }
        self._add_creation_info(packed_obj)
        return packed_obj

    @classmethod
    def from_json(cls, data: Dict) -> Multipolygon:
        """
        Convert a json dict to Multipolygon.

        :param data: Multipolygon in json format as a dict.
        :type data: dict
        :returns: Multipolygon from json.
        :rtype: :class:`~supervisely.geometry.multipolygon.Multipolygon`
        """
        if PARTS not in data:
            raise KeyError(f'"{PARTS}" field is required to create "Multipolygon" object.')
        if not isinstance(data[PARTS], list):
            raise TypeError(f'"{PARTS}" field must be a list.')

        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)

        parts = []
        for part in data[PARTS]:
            if EXTERIOR not in part:
                raise KeyError(f'Each "{PARTS}" element must contain "{EXTERIOR}" field.')
            exterior = row_col_list_to_points(part[EXTERIOR], flip_row_col_order=True)
            interior = [
                row_col_list_to_points(i, flip_row_col_order=True)
                for i in part.get(INTERIOR, [])
            ]
            parts.append(Polygon(exterior, interior))

        return cls(
            parts=parts,
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    def _transform(self, transform_fn):
        result = deepcopy(self)
        result._parts = [part._transform(transform_fn) for part in self._parts]
        return result

    def resize(self, in_size: Tuple[int, int], out_size: Tuple[int, int]) -> Multipolygon:
        return self._transform(lambda p: p.resize(in_size, out_size))

    def scale(self, factor: float) -> Multipolygon:
        return self._transform(lambda p: p.scale(factor))

    def translate(self, drow: int, dcol: int) -> Multipolygon:
        return self._transform(lambda p: p.translate(drow, dcol))

    def rotate(self, rotator) -> Multipolygon:
        return self._transform(lambda p: p.rotate(rotator))

    def fliplr(self, img_size: Tuple[int, int]) -> Multipolygon:
        return self._transform(lambda p: p.fliplr(img_size))

    def flipud(self, img_size: Tuple[int, int]) -> Multipolygon:
        return self._transform(lambda p: p.flipud(img_size))

    def crop(self, rect: Rectangle) -> List[Multipolygon]:
        """
        Crops current Multipolygon.

        :param rect: Rectangle to crop Multipolygon from.
        :type rect: :class:`~supervisely.geometry.rectangle.Rectangle`
        :returns: list with cropped Multipolygon or empty list
        :rtype: List[:class:`~supervisely.geometry.multipolygon.Multipolygon`]
        """
        cropped_parts = []
        for part in self._parts:
            cropped_parts.extend(part.crop(rect))
        return [Multipolygon(cropped_parts)] if len(cropped_parts) > 0 else []

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        for part in self._parts:
            part._draw_impl(bitmap, color, thickness, config=config)

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        for part in self._parts:
            part._draw_contour_impl(bitmap, color, thickness, config=config)

    @property
    def area(self) -> float:
        """
        Multipolygon area.

        :returns: sum of all polygon part areas
        :rtype: float
        """
        return sum(part.area for part in self._parts)

    def to_bbox(self) -> Rectangle:
        return Rectangle.from_geometries_list(self._parts)

    def approx_dp(self, epsilon: float) -> Multipolygon:
        return Multipolygon([part.approx_dp(epsilon) for part in self._parts])

    @classmethod
    def allowed_transforms(cls):
        """
        Returns the allowed transforms for the Multipolygon.
        """
        from supervisely.geometry.alpha_mask import AlphaMask
        from supervisely.geometry.any_geometry import AnyGeometry
        from supervisely.geometry.bitmap import Bitmap
        from supervisely.geometry.rectangle import Rectangle

        return [AnyGeometry, Rectangle, Bitmap, AlphaMask, Polygon]
