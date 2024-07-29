# coding: utf-8

# docs
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import supervisely as sly
from supervisely._utils import unwrap_if_numpy
from supervisely.geometry import validation
from supervisely.geometry.constants import EXTERIOR, INTERIOR, POINTS
from supervisely.imaging import image as sly_image
from supervisely.io.json import JsonSerializable


class PointLocation(JsonSerializable):
    """
    PointLocation in (row, col) position. :class:`PointLocation<PointLocation>` object is immutable.

    :param row: Position of PointLocation object on height.
    :type row: Union[int, float]
    :param col: Position of PointLocation object on width.
    :type col: Union[int, float]

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        row = 100
        col = 200
        loc = sly.PointLocation(row, col)
    """

    def __init__(self, row: Union[int, float], col: Union[int, float]):
        self._row = unwrap_if_numpy(row)
        self._col = unwrap_if_numpy(col)

        self._integer_coords = True
        if isinstance(self._row, float) or isinstance(self._col, float):
            self._integer_coords = False

    @property
    def row(self) -> Union[int, float]:
        """
        Position of PointLocation on height.

        :return: Height of PointLocation
        :rtype: Union[int, float]
        :Usage example:

         .. code-block:: python

            print(loc.row)
            # Output: 100
        """
        return self._row

    @property
    def rounded_row(self) -> int:
        """
        Rounded position of PointLocation on height.

        :return: Height of PointLocation
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            print(loc.row)
            # Output: 100
        """
        return round(self.row)

    @property
    def col(self) -> Union[int, float]:
        """
        Position of PointLocation on width.

        :return: Width of PointLocation
        :rtype: Union[int, float]
        :Usage example:

         .. code-block:: python

            print(loc.col)
            # Output: 200
        """
        return self._col

    @property
    def rounded_col(self) -> int:
        """
        Rounded position of PointLocation on width.

        :return: Width of PointLocation
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            print(loc.col)
            # Output: 200
        """
        return round(self.col)

    def to_json(self) -> Dict:
        """
        Convert the PointLocation to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            loc_json = loc.to_json()
            print(loc_json)
            # Output: {
            #    "points": {
            #        "exterior": [
            #            [
            #                200,
            #                100
            #            ]
            #        ],
            #        "interior": []
            #    }
            # }
        """
        if self._integer_coords:
            exterior = [[self.rounded_col, self.rounded_row]]
        else:
            exterior = [[self.col, self.row]]
        packed_obj = {POINTS: {EXTERIOR: exterior, INTERIOR: []}}
        return packed_obj

    @classmethod
    def from_json(cls, data: Dict) -> PointLocation:
        """
        Convert a json dict to PointLocation. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: PointLocation in json format as a dict.
        :type data: dict
        :return: PointLocation object
        :rtype: :class:`PointLocation<PointLocation>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            loc_json = {
                "points": {
                    "exterior": [
                        [
                            200,
                            100
                        ]
                    ],
                    "interior": []
                }
            }
            loc = sly.PointLocation.from_json(loc_json)
        """
        validation.validate_geometry_points_fields(data)
        exterior = data[POINTS][EXTERIOR]
        if len(exterior) != 1:
            raise ValueError(
                '"exterior" field must contain exactly one point to create "PointLocation" object.'
            )
        return cls(row=exterior[0][1], col=exterior[0][0])

    def scale(self, factor: float) -> PointLocation:
        """
        Scale current PointLocation.

        :param factor: Scale parameter.
        :type factor: float
        :return: PointLocation object
        :rtype: :class:`PointLocation<PointLocation>`

        :Usage Example:

         .. code-block:: python

            # Remember that PointLocation class object is immutable, and we need to assign new instance of PointLocation to a new variable
            scale_loc = loc.scale(0.75)
        """
        return self.scale_frow_fcol(factor, factor)

    def scale_frow_fcol(self, frow: float, fcol: float) -> PointLocation:
        """
        Calculates new parameters of PointLocation after scaling in horizontal and vertical.

        :param frow: Scale parameter for height.
        :type frow: float
        :param fcol: Scale parameter for width.
        :type fcol: float
        :return: PointLocation object
        :rtype: :class:`PointLocation<PointLocation>`

        :Usage Example:

         .. code-block:: python

            # Remember that PointLocation class object is immutable, and we need to assign new instance of PointLocation to a new variable
            loc_scale_rc = loc.scale_frow_fcol(0.1, 2.7)
        """
        if self._integer_coords:
            row, col = round(self.row * frow), round(self.col * fcol)
        else:
            row, col = self.row * frow, self.col * fcol
        return PointLocation(row=row, col=col)

    def translate(self, drow: int, dcol: int) -> PointLocation:
        """
        Translate current PointLocation object.

        :param drow: Horizontal shift.
        :type drow: int
        :param dcol: Vertical shift.
        :type dcol: int
        :return: PointLocation object
        :rtype: :class:`PointLocation<PointLocation>`

        :Usage Example:

         .. code-block:: python

            # Remember that PointLocation class object is immutable, and we need to assign new instance of PointLocation to a new variable
            translate_loc = loc.translate(150, 350)
        """
        if self._integer_coords:
            row, col = round(self.row + drow), round(self.col + dcol)
        else:
            row, col = self.row + drow, self.col + dcol
        return PointLocation(row=row, col=col)

    def rotate(self, rotator: sly.geometry.image_rotator.ImageRotator) -> PointLocation:
        """
        Rotates current PointLocation object.

        :param rotator: ImageRotator object for rotation.
        :type rotator: ImageRotator
        :return: PointLocation object
        :rtype: :class:`PointLocation<PointLocation>`

        :Usage Example:

         .. code-block:: python

            from supervisely.geometry.image_rotator import ImageRotator

            # Remember that PointLocation class object is immutable, and we need to assign new instance of PointLocation to a new variable
            height, width = 300, 400
            rotator = ImageRotator((height, width), 25)
            rotate_loc = loc.rotate(rotator)
        """
        return rotator.transform_point(self)

    def resize(self, in_size: Tuple[int, int], out_size: Tuple[int, int]) -> PointLocation:
        """
        Resize current PointLocation object.

        :param in_size: Input image size (height, width) to which belongs :class:`PointLocation<PointLocation>` object.
        :type in_size: Tuple[int, int]
        :param out_size: Desired output image size (height, width) to which belongs :class:`PointLocation<PointLocation>` object.
        :type out_size: Tuple[int, int]
        :return: PointLocation object
        :rtype: :class:`PointLocation<PointLocation>`

        :Usage Example:

         .. code-block:: python

            # Remember that PointLocation class object is immutable, and we need to assign new instance of PointLocation to a new variable
            in_height, in_width = 300, 400
            out_height, out_width = 600, 800
            resize_loc = loc.resize((in_height, in_width), (out_height, out_width))
        """
        new_size = sly_image.restore_proportional_size(in_size=in_size, out_size=out_size)
        frow = new_size[0] / in_size[0]
        fcol = new_size[1] / in_size[1]
        return self.scale_frow_fcol(frow=frow, fcol=fcol)

    def fliplr(self, img_size: Tuple[int, int]) -> PointLocation:
        """
        Flips current PointLocation object in horizontal.

        :param img_size: Input image size (height, width) to which belongs :class:`PointLocation<PointLocation>` object.
        :type img_size: Tuple[int, int]
        :return: PointLocation object
        :rtype: :class:`PointLocation<PointLocation>`

        :Usage Example:

         .. code-block:: python

            # Remember that PointLocation class object is immutable, and we need to assign new instance of PointLocation to a new variable
            height, width = 300, 400
            fliplr_loc = loc.fliplr((height, width))
        """
        if self._integer_coords:
            row, col = self.rounded_row, img_size[1] - self.rounded_col
        else:
            row, col = self.row, img_size[1] - self.col
        return PointLocation(row=row, col=col)

    def flipud(self, img_size: Tuple[int, int]) -> PointLocation:
        """
        Flips current PointLocation object in vertical.

        :param img_size: Input image size (height, width) to which belongs :class:`PointLocation<PointLocation>` object.
        :type img_size: Tuple[int, int]
        :return: PointLocation object
        :rtype: :class:`PointLocation<PointLocation>`

        :Usage Example:

         .. code-block:: python

            # Remember that PointLocation class object is immutable, and we need to assign new instance of PointLocation to a new variable
            height, width = 300, 400
            flipud_loc = loc.flipud((height, width))
        """
        if self._integer_coords:
            row, col = img_size[0] - self.rounded_row, self.rounded_col
        else:
            row, col = img_size[0] - self.row, self.col

        return PointLocation(row=row, col=col)

    def clone(self) -> PointLocation:
        """
        Makes a copy of the PointLocation object.

        :return: PointLocation object
        :rtype: :class:`PointLocation<PointLocation>`

        :Usage Example:

         .. code-block:: python

            # Remember that PointLocation class object is immutable, and we need to assign new instance of PointLocation to a new variable
            new_loc = loc.clone()
        """
        return deepcopy(self)


def _flip_row_col_order(coords):
    """ """
    if not all(len(x) == 2 for x in coords):
        raise ValueError(
            "Flipping row and column order values is only possible within tuples of 2 elements."
        )
    return [[y, x] for x, y in coords]


def _maybe_flip_row_col_order(coords, flip=False):
    """ """
    return _flip_row_col_order(coords) if flip else coords


def points_to_row_col_list(
    points: List[PointLocation], flip_row_col_order: Optional[bool] = False
) -> List[List[int]]:
    """
    Convert list of PointLocation objects to list of coords.

    :param points: List of PointLocation objects.
    :type points: List[PointLocation]
    :param flip_row_col_order: Flips row col coords if True.
    :type flip_row_col_order: bool, optional
    :return: List of coords
    :rtype: :class:`list`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            loc_1 = sly.PointLocation(100, 200)
            loc_2 = sly.PointLocation(300, 400)
            points_row_col = points_to_row_col_list([loc_1, loc_2])
            print(points_row_col)
            # Output: [[100, 200], [300, 400]]
    """
    return _maybe_flip_row_col_order(
        coords=[[p.row, p.col] for p in points], flip=flip_row_col_order
    )


def row_col_list_to_points(
    data: List[List[Union[int, float], Union[int, float]]],
    flip_row_col_order: Optional[bool] = False,
    do_round: Optional[bool] = False,
) -> List[PointLocation]:
    """
    Convert list of coords to list of PointLocation objects.

    :param data: List of coords.
    :type data: List[List[int, int]]
    :param flip_row_col_order: Flip row col coords if True.
    :type flip_row_col_order: bool, optional
    :param do_round: Round PointLocation params if True.
    :type do_round: bool, optional
    :return: List of PointLocation objects
    :rtype: :class:`List[PointLocation]`

        :Usage Example:

         .. code-block:: python

            row_1, col_1 = 100, 200
            row_2, col_2 = 300, 400
            coords = [(row_1, col_1), (row_2, col_2)]
            locs = row_col_list_to_points(coords)
    """

    def _maybe_round(v):
        return v if not do_round else round(v)

    return [
        PointLocation(row=_maybe_round(r), col=_maybe_round(c))
        for r, c in _maybe_flip_row_col_order(data, flip=flip_row_col_order)
    ]
