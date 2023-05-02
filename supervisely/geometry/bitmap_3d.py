# coding: utf-8

# docs
from typing import Optional, List, Tuple, Dict
from supervisely.geometry.geometry import Geometry
from supervisely.geometry import validation
from supervisely.geometry.constants import (
    EXTERIOR,
    POINTS,
    ORIGIN,
    DATA,
    GEOMETRY_SHAPE,
    GEOMETRY_TYPE,
    LABELER_LOGIN,
    UPDATED_AT,
    CREATED_AT,
    ID,
    CLASS_ID,
    BITMAP_3D,
)
from supervisely._utils import unwrap_if_numpy
from supervisely.io.json import JsonSerializable
import numpy as np
import io
import base64
import zlib
import cv2

from PIL import Image

if not hasattr(np, "bool"):
    np.bool = np.bool_


class PointLocation3D(JsonSerializable):
    """
    PointLocation3D in (row, col, tab) position. :class:`PointLocation3D<PointLocation3D>` object is immutable.

    :param row: Position of PointLocation3D object on height.
    :type row: int or float
    :param col: Position of PointLocation3D object on width.
    :type col: int or float
    :param tab: Position of PointLocation3D object on depth.
    :type tab: int or float

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        row = 100
        col = 200
        tab = 2
        loc = sly.PointLocation3D(row, col, tab)
    """

    def __init__(self, row: int, col: int, tab: int):
        self._row = round(unwrap_if_numpy(row))
        self._col = round(unwrap_if_numpy(col))
        self._tab = round(unwrap_if_numpy(tab))

    @property
    def row(self) -> int:
        """
        Position of PointLocation3D on height.

        :return: Height of PointLocation3D
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            print(loc.row)
            # Output: 100
        """
        return self._row

    @property
    def col(self) -> int:
        """
        Position of PointLocation3D on width.

        :return: Width of PointLocation3D
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            print(loc.col)
            # Output: 200
        """
        return self._col

    @property
    def tab(self) -> int:
        """
        Position of PointLocation3D on depth.

        :return: Depth of PointLocation3D
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            print(loc.tab)
            # Output: 2
        """
        return self._col

    def to_json(self) -> Dict:
        """
        Convert the PointLocation3D to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

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
            #        ]
            #    }
            # }
        """
        packed_obj = {POINTS: {EXTERIOR: [[self.col, self.row, self.tab]]}}
        return packed_obj

    @classmethod
    def from_json(cls, data: Dict):
        """
        Convert a json dict to PointLocation3D. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: PointLocation3D in json format as a dict.
        :type data: dict
        :return: PointLocation3D object
        :rtype: :class:`PointLocation3D<PointLocation3D>`
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
            loc = sly.PointLocation3D.from_json(loc_json)
        """
        validation.validate_geometry_points_fields(data)
        exterior = data[POINTS][EXTERIOR]
        if len(exterior) != 1:
            raise ValueError(
                '"exterior" field must contain exactly one point to create "PointLocation3D" object.'
            )
        return cls(row=exterior[0][1], col=exterior[0][0], tab=exterior[0][2])


class Bitmap3d(Geometry):
    """
    Bitmap 3D geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`Bitmap3d<Bitmap3d>` object is immutable.

    :param data: Bitmap 3D mask data. Must be a numpy array with only 2 unique values: [0, 1] or [0, 255] or [False, True].
    :type data: np.ndarray
    :param sly_id: Bitmap 3D ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which Bitmap 3D belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created Bitmap 3D.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when Bitmap 3D was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when Bitmap 3D was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :raises: :class:`ValueError`, if data is not bool or no pixels set to True in data
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create simple Bitmap 3D
        bitmap3d = np.zeros((3, 3, 3), dtype=np.bool_)
        bitmap3d[0:2, 0:2, 0:2] = True

        shape = sly.Bitmap3d(bitmap3d)

        print(shape.data)
        # Output:
        #    [[[ True  True False]
        #      [ True  True False]
        #      [False False False]]

        #     [[ True  True False]
        #      [ True  True False]
        #      [False False False]]

        #     [[False False False]
        #      [False False False]
        #      [False False False]]]
    """

    def __init__(
        self,
        data: np.ndarray,
        space: Optional[str] = None,
        space_origin: Optional[PointLocation3D] = None,
        space_directions: Optional[List[Tuple[float, float, float]]] = None,
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        super().__init__(
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

        if not isinstance(data, np.ndarray):
            raise TypeError('Bitmap 3D "data" argument must be numpy array object!')

        data_dims = len(data.shape)
        if data_dims != 3:
            raise ValueError(
                f'Bitmap 3D "data" argument must be a 3-dimensional numpy array. Instead got {data_dims} dimensions'
            )

        if data.dtype != np.bool:
            if list(np.unique(data)) not in [[0, 1], [0, 255]]:
                raise ValueError(
                    f"Bitmap 3D mask data values must be one of: [0 1], [0 255], [False True]. Instead got {np.unique(data)}."
                )

            if list(np.unique(data)) == [0, 1]:
                data = np.array(data, dtype=bool)
            elif list(np.unique(data)) == [0, 255]:
                data = np.array(data / 255, dtype=bool)

        self.data = data
        self.space = space
        self.space_origin = space_origin
        self.space_directions = space_directions

    @staticmethod
    def geometry_name():
        """geometry_name"""
        return "bitmap_3d"

    def to_json(self) -> Dict:
        """
        Convert the Bitmap3D to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            mask = np.array([[0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0],
                             [0, 1, 0, 1, 0],
                             [0, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0]], dtype=np.bool_)

            figure = sly.Bitmap(mask)
            figure_json = figure.to_json()
            print(json.dumps(figure_json, indent=4))
            # Output: {
            #    "bitmap": {
            #        "origin": [1, 1],
            #        "data": "eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6"
            #    },
            #    "shape": "bitmap",
            #    "geometryType": "bitmap"
            # }
        """
        res = {
            self._impl_json_class_name(): {
                ORIGIN: [2, 2, 2],
                # ORIGIN: [self.space_origin.col, self.space_origin.row, self.space_origin.tab],
                DATA: self.data_2_base64(self.data),
            },
            GEOMETRY_SHAPE: self.geometry_name(),
            GEOMETRY_TYPE: self.geometry_name(),
        }
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, json_data: Dict):
        """
        Convert a json dict to Bitmap3D. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Bitmap in json format as a dict.
        :type data: dict
        :return: Bitmap3D object
        :rtype: :class:`Bitmap3D<Bitmap3D>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            figure_json = {
                "bitmap": {
                    "origin": [1, 1],
                    "data": "eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6"
                },
                "shape": "bitmap_3d",
                "geometryType": "bitmap_3d"
            }

            figure = sly.Bitmap.from_json(figure_json)
        """
        if json_data == {}:
            return cls(data=np.zeros((3, 3, 3), dtype=np.bool_))

        json_root_key = cls._impl_json_class_name()
        if json_root_key not in json_data:
            raise ValueError(
                "Data must contain {} field to create MultichannelBitmap object.".format(
                    json_root_key
                )
            )

        if ORIGIN not in json_data[json_root_key] or DATA not in json_data[json_root_key]:
            raise ValueError(
                "{} field must contain {} and {} fields to create MultichannelBitmap object.".format(
                    json_root_key, ORIGIN, DATA
                )
            )

        col, row, tab = json_data[json_root_key][ORIGIN]
        # data = cls.base64_2_data(json_data[json_root_key][DATA])
        data = json_data[json_root_key][DATA]

        labeler_login = json_data.get(LABELER_LOGIN, None)
        updated_at = json_data.get(UPDATED_AT, None)
        created_at = json_data.get(CREATED_AT, None)
        sly_id = json_data.get(ID, None)
        class_id = json_data.get(CLASS_ID, None)
        return cls(
            data=data,
            space_origin=PointLocation3D(row=row, col=col, tab=tab),
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

    @classmethod
    def _impl_json_class_name(cls):
        """_impl_json_class_name"""
        return BITMAP_3D

    @staticmethod
    def data_2_base64(mask: np.ndarray) -> str:
        mask_8bit = mask.astype(np.uint8) * 255
        mask_8bit = np.transpose(mask_8bit, (1, 2, 0))  # Convert to (H, W, D) order
        img_pil = Image.fromarray(mask_8bit, mode="RGB")
        bytes_io = io.BytesIO()
        img_pil.save(bytes_io, format="PNG", transparency=(0, 0, 0))
        bytes_enc = bytes_io.getvalue()
        return base64.b64encode(zlib.compress(bytes_enc)).decode("utf-8")

    def base64_2_data_3d(s: str) -> np.ndarray:
        z = zlib.decompress(base64.b64decode(s))
        n = np.frombuffer(z, np.uint8)

        imdecoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)

        if len(imdecoded.shape) == 3 and imdecoded.shape[2] >= 4:
            alpha_channel = imdecoded[:, :, 3]
            rgb_channels = cv2.split(imdecoded[:, :, :3])
            bitmap_2d = np.concatenate(rgb_channels, axis=1)
            bitmap_3d = np.expand_dims(bitmap_2d, axis=2) * (alpha_channel != 0)
            for i in range(1, imdecoded.shape[2] - 1):
                channels = cv2.split(imdecoded[:, :, i : i + 4])
                bitmap_2d = np.concatenate(channels[:3], axis=1)
                bitmap_3d_slice = np.expand_dims(bitmap_2d, axis=2) * (channels[3] != 0)
                bitmap_3d = np.concatenate([bitmap_3d, bitmap_3d_slice], axis=2)
            bitmap_2d = np.concatenate(imdecoded[:, :, -3:], axis=1)
            bitmap_3d_slice = np.expand_dims(bitmap_2d, axis=2) * (imdecoded[:, :, -1] != 0)
            bitmap_3d = np.concatenate([bitmap_3d, bitmap_3d_slice], axis=2)
            return bitmap_3d.astype(bool)
        else:
            raise RuntimeError("Wrong internal mask format.")
