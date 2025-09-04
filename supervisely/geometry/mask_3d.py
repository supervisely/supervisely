# coding: utf-8

# docs
from __future__ import annotations

import base64
import gzip
import tempfile
from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from supervisely import logger
from supervisely._utils import unwrap_if_numpy
from supervisely.geometry.constants import (
    CLASS_ID,
    CREATED_AT,
    DATA,
    GEOMETRY_SHAPE,
    GEOMETRY_TYPE,
    ID,
    LABELER_LOGIN,
    MASK_3D,
    SPACE,
    SPACE_DIRECTIONS,
    SPACE_ORIGIN,
    UPDATED_AT,
)
from supervisely.geometry.geometry import Geometry
from supervisely.io.fs import get_file_ext, get_file_name, remove_dir
from supervisely.io.json import JsonSerializable

if not hasattr(np, "bool"):
    np.bool = np.bool_


class PointVolume(JsonSerializable):
    """
    PointVolume (x, y, z) determines position of Mask3D. It locates the first sample.
    :class:`PointVolume<PointVolume>` object is immutable.

    :param x: Position of PointVolume object on X-axis.
    :type x: int or float
    :param y: Position of PointVolume object on Y-axis.
    :type y: int or float
    :param z: Position of PointVolume object on Z-axis.
    :type z: int or float
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        x = 100
        y = 200
        z = 2
        loc = sly.PointVolume(x, y, z)
    """

    def __init__(self, x: Union[int, float], y: Union[int, float], z: Union[int, float]):
        self._x = round(unwrap_if_numpy(x))
        self._y = round(unwrap_if_numpy(y))
        self._z = round(unwrap_if_numpy(z))

    @property
    def x(self) -> int:
        """
        Position of PointVolume on X-axis.

        :return: X of PointVolume
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            print(loc.x)
            # Output: 100
        """
        return self._x

    @property
    def y(self) -> int:
        """
        Position of PointVolume on Y-axis.

        :return: Y of PointVolume
        :rtype: :class:`int`

        :Usage example:

         .. code-block:: python

            print(loc.y)
            # Output: 200
        """
        return self._y

    @property
    def z(self) -> int:
        """
        Position of PointVolume on Z-axis.

        :return: Z of PointVolume
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            print(loc.z)
            # Output: 2
        """
        return self._z

    def to_json(self) -> Dict:
        """
        Convert the PointVolume to a json dict.

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

        packed_obj = {SPACE_ORIGIN: [self.x, self.y, self.z]}
        return packed_obj

    @classmethod
    def from_json(cls, packed_obj) -> PointVolume:
        """
        Convert a json dict to PointVolume.

        :param data: PointVolume in json format as a dict.
        :type data: dict
        :return: PointVolume object
        :rtype: :class:`PointVolume<PointVolume>`
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

            loc = sly.PointVolume.from_json(loc_json)
        """
        return cls(
            x=packed_obj["space_origin"][0],
            y=packed_obj["space_origin"][1],
            z=packed_obj["space_origin"][2],
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
    :param volume_header: NRRD header dictionary. Optional.
    :type volume_header: dict, optional
    :param convert_to_ras: If True, converts the mask to RAS orientation. Default is True.
    :type convert_to_ras: bool, optional
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
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
        volume_header: Optional[Dict] = None,
        convert_to_ras: bool = True,
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
        self._space_origin = None
        self._space = None
        self._space_directions = None

        if volume_header is not None:
            self.set_volume_space_meta(volume_header)
            if self.space is not None and self.space != "right-anterior-superior":
                if convert_to_ras:
                    self.orient_ras()
                else:
                    logger.debug(
                        "Mask3D is not in RAS orientation. It is recommended to use RAS orientation for 3D masks."
                    )

    @property
    def space_origin(self) -> Optional[List[float]]:
        """
        Get the space origin of the Mask3D as a list of floats.

        :return: Space origin of the Mask3D.
        :rtype: List[float] or None
        """
        if self._space_origin is not None:
            return [self._space_origin.x, self._space_origin.y, self._space_origin.z]
        return None

    @space_origin.setter
    def space_origin(self, value: Union[PointVolume, List[float], np.array]):
        """
        Set the space origin of the Mask3D.

        :param value: Space origin of the Mask3D. If provided as a list or array, it should contain 3 floats in the order [x, y, z].
        :type value: :class:`PointVolume<PointVolume>` or List[float]
        """
        if isinstance(value, PointVolume):
            self._space_origin = value
        elif isinstance(value, list) and len(value) == 3:
            self._space_origin = PointVolume(x=value[0], y=value[1], z=value[2])
        elif isinstance(value, np.ndarray) and value.shape == (3,):
            self._space_origin = PointVolume(x=value[0], y=value[1], z=value[2])
        else:
            raise ValueError("Space origin must be a PointVolume or a list of 3 floats.")

    @property
    def space(self) -> Optional[str]:
        """
        Get the space of the Mask3D.

        :return: Space of the Mask3D.
        :rtype: :class:`str`
        """
        return self._space

    @space.setter
    def space(self, value: str):
        """
        Set the space of the Mask3D.

        :param value: Space of the Mask3D.
        :type value: str
        """
        if not isinstance(value, str):
            raise ValueError("Space must be a string.")
        self._space = value

    @property
    def space_directions(self) -> Optional[List[List[float]]]:
        """
        Get the space directions of the Mask3D.

        :return: Space directions of the Mask3D.
        :rtype: :class:`List[List[float]]`
        """
        return self._space_directions

    @space_directions.setter
    def space_directions(self, value: Union[List[List[float]], np.ndarray]):
        """
        Set the space directions of the Mask3D.

        :param value: Space directions of the Mask3D. Should be a 3x3 array-like structure.
        :type value: List[List[float]] or np.ndarray
        """
        if isinstance(value, np.ndarray):
            if value.shape != (3, 3):
                raise ValueError("Space directions must be a 3x3 array.")
            self._space_directions = value.tolist()
        elif (
            isinstance(value, list)
            and len(value) == 3
            and all(isinstance(row, (list, np.ndarray)) and len(row) == 3 for row in value)
        ):
            self._space_directions = [list(row) for row in value]
        else:
            raise ValueError("Space directions must be a 3x3 array or list of lists.")

    @staticmethod
    def geometry_name():
        """Return geometry name"""
        return "mask_3d"

    @staticmethod
    def from_file(figure, file_path: str):
        """
        Load figure geometry from file.

        :param figure: Spatial figure
        :type figure: VolumeFigure
        :param file_path: Path to nrrd file with data
        :type file_path: str
        """
        mask3d = Mask3D.create_from_file(file_path)
        figure._set_3d_geometry(mask3d)
        path_without_filename = "/".join(file_path.split("/")[:-1])
        remove_dir(path_without_filename)

    @classmethod
    def create_from_file(cls, file_path: str) -> Mask3D:
        """
        Creates Mask3D geometry from file.

        :param file_path: Path to nrrd file with data
        :type file_path: str
        """
        from supervisely.volume.volume import read_nrrd_serie_volume_np

        mask3d_data, meta = read_nrrd_serie_volume_np(file_path)
        direction = np.array(meta["directions"]).reshape(3, 3)
        spacing = np.array(meta["spacing"])
        space_directions = (direction.T * spacing[:, None]).tolist()
        mask3d_header = {
            "space": "right-anterior-superior",
            "space directions": space_directions,
            "space origin": meta.get("origin", None),
        }

        geometry = cls(data=mask3d_data, volume_header=mask3d_header)

        fields_to_check = ["space", "space_directions", "space_origin"]
        if any([getattr(geometry, value) is None for value in fields_to_check]):
            header_keys = ["'space'", "'space directions'", "'space origin'"]
            logger.debug(
                f"The Mask3D geometry created from the file '{file_path}' doesn't contain optional space attributes that have similar names to {', '.join(header_keys)}. To set the values for these attributes, you can use information from the Volume associated with this figure object."
            )
        return geometry

    @classmethod
    def from_bytes(cls, geometry_bytes: bytes) -> Mask3D:
        """
        Create a Mask3D geometry object from bytes.

        :param geometry_bytes: NRRD file represented as bytes.
        :type geometry_bytes: bytes
        :return: A Mask3D geometry object.
        :rtype: Mask3D
        """
        with tempfile.NamedTemporaryFile(delete=True, suffix=".nrrd") as temp_file:
            temp_file.write(geometry_bytes)
            return cls.create_from_file(temp_file.name)

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

            figure = sly.Mask3D(mask)
            figure_json = figure.to_json()
            print(json.dumps(figure_json, indent=4))
            # Output: {
            #    "mask_3d": {
            #        "data": "eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6",
            #    },
            #    "shape": "mask_3d",
            #    "geometryType": "mask_3d"
            # }
        """
        res = {
            self._impl_json_class_name(): {
                DATA: self.data_2_base64(self.data),
            },
            GEOMETRY_SHAPE: self.name(),
            GEOMETRY_TYPE: self.name(),
        }

        if self.space_origin:
            res[f"{self._impl_json_class_name()}"][f"{SPACE_ORIGIN}"] = self.space_origin

        if self.space:
            res[f"{self._impl_json_class_name()}"][f"{SPACE}"] = self.space

        if self.space_directions:
            res[f"{self._impl_json_class_name()}"][f"{SPACE_DIRECTIONS}"] = self.space_directions

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

        if DATA not in json_data[json_root_key]:
            raise ValueError(
                "{} field must contain {} field to create Mask3D object.".format(
                    json_root_key, DATA
                )
            )

        data = cls.base64_2_data(json_data[json_root_key][DATA])

        labeler_login = json_data.get(LABELER_LOGIN, None)
        updated_at = json_data.get(UPDATED_AT, None)
        created_at = json_data.get(CREATED_AT, None)
        sly_id = json_data.get(ID, None)
        class_id = json_data.get(CLASS_ID, None)

        header = {}

        space_origin = json_data[json_root_key].get(SPACE_ORIGIN, None)
        if space_origin is not None:
            header["space origin"] = space_origin

        space = json_data[json_root_key].get(SPACE, None)
        if space is not None:
            header["space"] = space

        space_directions = json_data[json_root_key].get(SPACE_DIRECTIONS, None)
        if space_directions is not None:
            header["space directions"] = space_directions

        return cls(
            data=data.astype(np.bool_),
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
            volume_header=header,
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

            address = 'https://app.supervisely.com/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            meta_json = api.project.get_meta(PROJECT_ID)
            meta = sly.ProjectMeta.from_json(meta_json)

            ann_json = api.volume.annotation.download_bulk(DATASET_ID, [VOLUME_ID])

            figure_id = ann_json[0]["spatialFigures"][0]["id"]
            path_for_mesh = f"meshes/{figure_id}.nrrd"
            api.volume.figure.download_stl_meshes([figure_id], [path_for_mesh])

            mask3d_data, _ = sly.volume.volume.read_nrrd_serie_volume_np(path_for_mesh)
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
        try:
            data = np.frombuffer(data_bytes, dtype=np.uint8).reshape(shape)
        except ValueError:
            logger.warn(
                "Can't reshape array with 'dtype=np.uint8'. Will try to automatically convert 'dtype=np.int16' to 'np.uint8' and reshape"
            )
            data = np.frombuffer(data_bytes, dtype=np.int16)
            data = np.clip(data, 0, 1).astype(np.uint8)
            data = data.reshape(shape)
            logger.debug("Converted successfully!")
        return data

    def add_mask_2d(
        self,
        mask_2d: np.ndarray,
        plane_name: Literal["axial", "sagittal", "coronal"],
        slice_index: int,
        origin: Optional[List[int]] = None,
    ):
        """
        Draw a 2D mask on a 3D Mask.

        :param mask_2d: 2D array with a flat mask.
        :type mask_2d: np.ndarray
        :param plane_name: Name of the plane: "axial", "sagittal", "coronal".
        :type plane_name: str
        :param slice_index: Slice index of the volume figure.
        :type slice_index: int
        :param origin: (row, col) position. The top-left corner of the mask is located on the specified slice (optional).
        :type origin: Optional[List[int]], NoneType
        """

        from supervisely.volume_annotation.plane import Plane

        Plane.validate_name(plane_name)

        mask_2d = np.fliplr(mask_2d)
        mask_2d = np.rot90(mask_2d, 1, (1, 0))

        if plane_name == Plane.AXIAL:
            new_shape = self.data.shape[:2]
        elif plane_name == Plane.SAGITTAL:
            new_shape = self.data.shape[1:]
        elif plane_name == Plane.CORONAL:
            new_shape = self.data.shape[::2]

        if origin:
            x, y = origin
            # pylint: disable=possibly-used-before-assignment
            new_mask = np.zeros(new_shape, dtype=mask_2d.dtype)
            new_mask[x : x + mask_2d.shape[0], y : y + mask_2d.shape[1]] = mask_2d

        if plane_name == Plane.AXIAL:
            self.data[:, :, slice_index] = new_mask
        elif plane_name == Plane.SAGITTAL:
            self.data[slice_index, :, :] = new_mask
        elif plane_name == Plane.CORONAL:
            self.data[:, slice_index, :] = new_mask

    @staticmethod
    def _bytes_from_nrrd(path: str) -> Tuple[str, bytes]:
        """
        Read geometry from a file as bytes.

        The NRRD file must be named with a hexadecimal UUID value. Only NRRD files are supported.

        :param path: Path to the NRRD file containing geometry.
        :type path: str
        :return: A tuple containing the key hex value and geometry bytes, or (None, None) if the file is not found.
        :rtype: Tuple[str, bytes]
        """

        if get_file_ext(path) == ".nrrd":
            key = get_file_name(path)
            with open(path, "rb") as file:
                geometry_bytes = file.read()
            return key, geometry_bytes
        else:
            return None, None

    @staticmethod
    def _bytes_from_nrrd_batch(paths: List[str]) -> Dict[str, bytes]:
        """
        Read geometries from multiple files as bytes and map them to figure UUID hex values in a dictionary.

        The NRRD files must be named with a hexadecimal UUID value. Only NRRD files are supported.

        :param paths: Paths to the NRRD files containing geometry.
        :type paths: List[str]
        :return: A dictionary mapping figure UUID hex values to their respective geometries.
        :rtype: Dict[str, bytes]
        """
        geometries_dict = {}
        for path in paths:
            key, geometry_bytes = Mask3D._bytes_from_nrrd(path)
            if key is None and geometry_bytes is None:
                continue
            geometries_dict[key] = geometry_bytes
        return geometries_dict

    def set_volume_space_meta(self, header: Dict):
        """
        Set space, space directions, and space origin attributes from a NRRD header dictionary.

        :param header: NRRD header dictionary.
        :type header: dict
        """
        if "space" in header:
            self.space = header["space"]
        if "space directions" in header:
            self.space_directions = header["space directions"]
        if "space origin" in header:
            self.space_origin = PointVolume(
                x=header["space origin"][0],
                y=header["space origin"][1],
                z=header["space origin"][2],
            )

    def create_header(self) -> OrderedDict:
        """
        Create header for encoding Mask3D to NRRD bytes

        :return: Header for NRRD file
        :rtype: OrderedDict
        """
        header = OrderedDict()
        if self.space is not None:
            header["space"] = self.space
        if self.space_directions is not None:
            header["space directions"] = self.space_directions
        if self.space_origin is not None:
            header["space origin"] = self.space_origin
        return header

    def orient_ras(self) -> None:
        """
        Transforms the mask data and updates spatial metadata (origin, directions, spacing)
        to align with the RAS coordinate system using SimpleITK.

        :rtype: None
        """
        import SimpleITK as sitk

        from supervisely.volume.volume import _sitk_image_orient_ras

        # Convert bool data to uint8 for SimpleITK compatibility
        data_for_sitk = self.data.astype(np.uint8) if self.data.dtype == np.bool_ else self.data
        sitk_volume = sitk.GetImageFromArray(data_for_sitk)
        if self.space_origin is not None:
            sitk_volume.SetOrigin(self.space_origin)
        if self.space_directions is not None:
            # Convert space directions to spacing and direction
            space_directions = np.array(self.space_directions)
            spacing = np.linalg.norm(space_directions, axis=1)
            direction = space_directions / spacing[:, np.newaxis]
            sitk_volume.SetSpacing(spacing)
            sitk_volume.SetDirection(direction.flatten())

        sitk_volume = _sitk_image_orient_ras(sitk_volume)

        # Extract transformed data and update object
        self.data = sitk.GetArrayFromImage(sitk_volume)
        new_direction = np.array(sitk_volume.GetDirection()).reshape(3, 3)
        new_spacing = np.array(sitk_volume.GetSpacing())
        new_space_directions = (new_direction.T * new_spacing[:, None]).tolist()
        new_header = {
            "space": "right-anterior-superior",
            "space directions": new_space_directions,
            "space origin": sitk_volume.GetOrigin(),
        }
        self.set_volume_space_meta(new_header)
