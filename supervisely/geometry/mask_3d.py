# coding: utf-8

# docs
from __future__ import annotations
from typing import Optional, Union, List, Tuple, Dict
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.constants import (
    SPACE_ORIGIN,
    DATA,
    GEOMETRY_SHAPE,
    GEOMETRY_TYPE,
    LABELER_LOGIN,
    UPDATED_AT,
    CREATED_AT,
    ID,
    CLASS_ID,
    MASK_3D,
)
from supervisely._utils import unwrap_if_numpy
from supervisely.io.json import JsonSerializable
import numpy as np
import base64
import gzip


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

    def __init__(self, row: Union[int, float], col: Union[int, float], tab: Union[int, float]):
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
        return self._tab

    def to_json(self) -> Dict:
        """
        Convert the PointLocation3D to a json dict.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            loc_json = loc.to_json()
            print(loc_json)
            # Output: {
            #           "space_origin": [
            #                            200,
            #                            200,
            #                            100
            #                           ]
            #         }

        """

        packed_obj = {SPACE_ORIGIN: [self.col, self.row, self.tab]}
        return packed_obj

    @classmethod
    def from_json(cls, packed_obj) -> PointLocation3D:
        """
        Convert a json dict to PointLocation3D.

        :param data: PointLocation3D in json format as a dict.
        :type data: dict
        :return: PointLocation3D object
        :rtype: :class:`PointLocation3D<PointLocation3D>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            loc_json = {
                "space_origin": [
                                200,
                                200,
                                100,
                                ]
                        }

            loc = sly.PointLocation3D.from_json(loc_json)
        """
        return cls(
            row=packed_obj["space_origin"][0],
            col=packed_obj["space_origin"][1],
            tab=packed_obj["space_origin"][2],
        )


class Mask3D(Geometry):
    """
    Mask 3D geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`Mask3D<Mask3D>` object is immutable.

    :param data: Mask 3D mask data. Must be a numpy array with only 2 unique values: [0, 1] or [0, 255] or [False, True].
    :type data: np.ndarray
    :param sly_id: Mask 3D ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which Mask 3D belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created Mask 3D.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when Mask 3D was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when Mask 3D was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :raises: :class:`ValueError`, if data is not bool or no pixels set to True in data
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create simple Mask 3D
        mask3d = np.zeros((3, 3, 3), dtype=np.bool_)
        mask3d[0:2, 0:2, 0:2] = True

        shape = sly.Mask3D(mask3d)

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
            raise TypeError('Mask 3D "data" argument must be numpy array object!')

        data_dims = len(data.shape)
        if data_dims != 3:
            raise ValueError(
                f'Mask 3D "data" argument must be a 3-dimensional numpy array. Instead got {data_dims} dimensions'
            )

        if data.dtype != np.bool:
            if list(np.unique(data)) not in [[0, 1], [0, 255]]:
                raise ValueError(
                    f"Mask 3D mask data values must be one of: [0 1], [0 255], [False True]. Instead got {np.unique(data)}."
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
        return "mask_3d"

    def to_json(self) -> Dict:
        """
        Convert the Mask 3D to a json dict.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            mask = np.array([[[1 1 0]
                              [1 1 0]
                              [0 0 0]]
                             [[1 1 0]
                              [1 1 0]
                              [0 0 0]]
                             [[0 0 0]
                              [0 0 0]
                              [0 0 0]]], dtype=np.bool_)

            location = sly.PointLocation3D(1, 1, 1)

            figure = sly.Mask3D(mask, space_origin=location)
            figure_json = figure.to_json()
            print(json.dumps(figure_json, indent=4))
            # Output: {
            #    "mask_3d": {
            #        "data": "eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6",
            #        "space_origin": [1, 1, 1],
            #    },
            #    "shape": "mask_3d",
            #    "geometryType": "mask_3d"
            # }
        """
        res = {
            self._impl_json_class_name(): {
                SPACE_ORIGIN: [
                    self.space_origin.col,
                    self.space_origin.row,
                    self.space_origin.tab,
                ],
                DATA: self.data_2_base64(self.data),
            },
            GEOMETRY_SHAPE: self.geometry_name(),
            GEOMETRY_TYPE: self.geometry_name(),
        }
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, json_data: Dict) -> Mask3D:
        """
        Convert a json dict to Mask 3D.

        :param data: Mask in json format as a dict.
        :type data: dict
        :return: Mask3D object
        :rtype: :class:`Mask3D<Mask3D>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            figure_json = {
                "mask_3d": {
                    "data": "eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6",
                    "space_origin": [1, 1, 1],
                },
                "shape": "mask_3d",
                "geometryType": "mask_3d"
            }

            figure = sly.Mask3D.from_json(figure_json)
        """
        if json_data == {}:
            return cls(data=np.zeros((3, 3, 3), dtype=np.bool_))

        json_root_key = cls._impl_json_class_name()
        if json_root_key not in json_data:
            raise ValueError(
                "Data must contain {} field to create Mask3D object.".format(json_root_key)
            )

        if SPACE_ORIGIN not in json_data[json_root_key] or DATA not in json_data[json_root_key]:
            raise ValueError(
                "{} field must contain {} and {} fields to create Mask3D object.".format(
                    json_root_key, SPACE_ORIGIN, DATA
                )
            )

        col, row, tab = json_data[json_root_key][SPACE_ORIGIN]
        data = cls.base64_2_data(json_data[json_root_key][DATA])

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
        return MASK_3D

    @staticmethod
    def data_2_base64(data: np.ndarray) -> str:
        """
        Convert numpy array to base64 encoded string.

        :param mask: Bool numpy array.
        :type mask: np.ndarray
        :return: Base64 encoded string
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import os
            import nrrd

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            meta_json = api.project.get_meta(PROJECT_ID)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_json = api.volume.annotation.download_bulk(DATASET_ID, [VOLUME_ID])

            figure_id = ann_json[0]["spatialFigures"][0]["id"]
            path_for_mesh = f"meshes/{figure_id}.nrrd"
            api.volume.figure.download_stl_meshes([figure_id], [path_for_mesh])

            mask3d_data, _ = nrrd.read(path_for_mesh)
            encoded_string = sly.Mask3D.data_2_base64(mask3d_data)

            print(encoded_string)
            # 'H4sIAGWoWmQC/zPWMdYxrmFkZAAiIIAz4AAAE56ciyEAAAA='
        """
        shape_str = ",".join(str(dim) for dim in data.shape)
        data_str = data.tostring().decode("utf-8")
        combined_str = f"{shape_str}|{data_str}"
        compressed_string = gzip.compress(combined_str.encode("utf-8"))
        encoded_string = base64.b64encode(compressed_string).decode("utf-8")
        return encoded_string

    @staticmethod
    def base64_2_data(encoded_string: str) -> np.ndarray:
        """
        Convert base64 encoded string to numpy array.

        :param s: Input base64 encoded string.
        :type s: str
        :return: Bool numpy array
        :rtype: :class:`np.ndarray`
        :Usage example:

         .. code-block:: python

              import supervisely as sly

              encoded_string = 'H4sIAGWoWmQC/zPWMdYxrmFkZAAiIIAz4AAAE56ciyEAAAA='
              figure_data = sly.Mask3D.base64_2_data(encoded_string)
              print(figure_data)
              # [[[1 1 0]
              #   [1 1 0]
              #   [0 0 0]]
              #  [[1 1 0]
              #   [1 1 0]
              #   [0 0 0]]
              #  [[0 0 0]
              #   [0 0 0]
              #   [0 0 0]]]
        """
        compressed_bytes = base64.b64decode(encoded_string)
        decompressed_string = gzip.decompress(compressed_bytes).decode("utf-8")
        shape_str, data_str = decompressed_string.split("|")
        shape = tuple(int(dim) for dim in shape_str.split(","))
        data_bytes = data_str.encode("utf-8")
        data = np.frombuffer(data_bytes, dtype=np.uint8).reshape(shape)
        return data
