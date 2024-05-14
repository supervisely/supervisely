# coding: utf-8

# docs
from __future__ import annotations

import base64
import io
import re
import zlib
from distutils.version import StrictVersion
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image
from requests.structures import CaseInsensitiveDict
from requests_toolbelt import MultipartDecoder, MultipartEncoder

from supervisely import logger
from supervisely._utils import batched
from supervisely.api.module_api import ApiField
from supervisely.geometry.bitmap import (
    Bitmap,
    SkeletonizeMethod,
    _find_mask_tight_bbox,
    resize_origin_and_bitmap,
)
from supervisely.geometry.bitmap_base import BitmapBase, resize_origin_and_bitmap
from supervisely.geometry.constants import ALPHA_MASK
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.point_location import PointLocation, row_col_list_to_points
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging import image as sly_image
from supervisely.imaging.image import read
from supervisely.io.network_exceptions import (
    process_requests_exception,
    process_unhandled_request,
)


class AlphaMask(Bitmap):
    """
    AlphaMask geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`AlphaMask<AlphaMask>` object is immutable.

    :param data: AlphaMask mask data. Must be a numpy array with values in range [0, 255].
    :type data: np.ndarray
    :param origin: :class:`PointLocation<supervisely.geometry.point_location.PointLocation>`: top, left corner of AlphaMask. Position of the AlphaMask within image.
    :type origin: PointLocation, optional
    :param sly_id: AlphaMask ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which AlphaMask belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created AlphaMask.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when AlphaMask was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when AlphaMask was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :param extra_validation: If True, additional validation is performed. Throws a ValueError if values of the data are not in the range [0, 255]. If True it will affect performance.
    :type extra_validation: bool, optional
    :raises: :class:`ValueError`, if data values are not in the range [0, 255].
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create simple alpha mask:
        mask = np.array([[0, 0, 0, 0, 0],
                         [0, 50, 50, 50, 0],
                         [0, 50, 0, 50, 0],
                         [0, 50, 50, 50, 0],
                         [0, 0, 0, 0, 0]], dtype=np.uint8)

        figure = sly.AlphaMask(mask)

        origin = figure.origin.to_json()
        print(json.dumps(origin, indent=4))
        # Output: {
        #     "points": {
        #         "exterior": [
        #             [
        #                 1,
        #                 1
        #             ]
        #         ],
        #         "interior": []
        #     }

        # Create alpha mask from PNG image:
        img = sly.imaging.image.read(os.path.join(os.getcwd(), 'black_white.png'))
        mask = img[:, :, 3]
        figure = sly.AlphaMask(mask)

     .. image:: https://i.imgur.com/2L3HRPs.jpg
    """

    @staticmethod
    def geometry_name():
        """geometry_name"""
        return "alpha_mask"

    def __init__(
        self,
        data: np.ndarray,
        origin: Optional[PointLocation] = None,
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[int] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
        extra_validation: Optional[bool] = True,
    ):
        if data.dtype != np.uint8:
            if data.dtype == np.bool:
                data = data.astype(np.uint8) * 225
            else:
                data = np.array(data, dtype=np.uint8)
            if extra_validation:
                if not np.all(np.isin(data, range(256))):
                    max_val = np.max(data)
                    min_val = np.min(data)
                    if max_val > 255 or min_val < 0:
                        raise ValueError(
                            f"Alpha mask data values must be in range [0, 255]. Instead got min: {min_val}, max: {max_val}."
                        )

        # Call base constructor first to do the basic dimensionality checks.
        BitmapBase.__init__(
            self,
            data=data,
            origin=origin,
            expected_data_dims=[2, 3],
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

        if not np.any(data):
            raise ValueError(
                "Creating a alpha mask with an empty mask (no pixels set to True) is not supported."
            )
        data_tight_bbox = _find_mask_tight_bbox(self._data)
        self._origin = self._origin.translate(drow=data_tight_bbox.top, dcol=data_tight_bbox.left)
        self._data = data_tight_bbox.get_cropped_numpy_slice(self._data)

    def rotate(self, rotator: ImageRotator) -> AlphaMask:
        """
        Rotates current AlphaMask.

        :param rotator: :class:`ImageRotator<supervisely.geometry.image_rotator.ImageRotator>` for Bitamp rotation.
        :type rotator: ImageRotator
        :return: AlphaMask object
        :rtype: :class:`AlphaMask<AlphaMask>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.image_rotator import ImageRotator

            height, width = ann.img_size
            rotator = ImageRotator((height, width), 25)
            # Remember that AlphaMask class object is immutable, and we need to assign new instance of AlphaMask to a new variable
            rotate_figure = figure.rotate(rotator)
        """
        full_img_mask = np.zeros(rotator.src_imsize, dtype=np.uint8)
        self.draw(full_img_mask, 255)
        # TODO this may break for one-pixel masks (it can disappear during rotation). Instead, rotate every pixel
        #  individually and set it in the resulting bitmap.
        new_mask = rotator.rotate_img(full_img_mask, use_inter_nearest=True)
        return AlphaMask(data=new_mask[:, :, :3])

    def crop(self, rect: Rectangle) -> List[AlphaMask]:
        """
        Crops current AlphaMask.

        :param rect: Rectangle object for cropping.
        :type rect: Rectangle
        :return: List of AlphaMasks
        :rtype: :class:`List[AlphaMask]<supervisely.geometry.alpha_mask.AlphaMask>`

        :Usage Example:

         .. code-block:: python

            crop_figures = figure.crop(sly.Rectangle(1, 1, 300, 350))
        """
        maybe_cropped_bbox = self.to_bbox().crop(rect)
        if len(maybe_cropped_bbox) == 0:
            return []
        else:
            [cropped_bbox] = maybe_cropped_bbox
            cropped_bbox_relative = cropped_bbox.translate(
                drow=-self.origin.row, dcol=-self.origin.col
            )
            cropped_mask = cropped_bbox_relative.get_cropped_numpy_slice(self._data)
            if not np.any(cropped_mask):
                return []
            return [
                AlphaMask(
                    data=cropped_mask,
                    origin=PointLocation(row=cropped_bbox.top, col=cropped_bbox.left),
                )
            ]

    def resize(self, in_size: Tuple[int, int], out_size: Tuple[int, int]) -> AlphaMask:
        """
        Resizes current AlphaMask.

        :param in_size: Input image size (height, width) to which AlphaMask belongs.
        :type in_size: Tuple[int, int]
        :param out_size: Output image size (height, width) to which AlphaMask belongs.
        :type out_size: Tuple[int, int]
        :return: AlphaMask object
        :rtype: :class:`AlphaMask<AlphaMask>`

        :Usage Example:

         .. code-block:: python

            in_height, in_width = 800, 1067
            out_height, out_width = 600, 800
            # Remember that AlphaMask class object is immutable, and we need to assign new instance of AlphaMask to a new variable
            resize_figure = figure.resize((in_height, in_width), (out_height, out_width))
        """
        scaled_origin, scaled_data = resize_origin_and_bitmap(
            self._origin, self._data.astype(np.uint8), in_size, out_size
        )
        # TODO this might break if a sparse mask is resized too thinly. Instead, resize every pixel individually and set
        #  it in the resulting bitmap.
        return AlphaMask(data=scaled_data.astype(np.uint8), origin=scaled_origin)

    # def set_data(self, data: np.ndarray) -> None:  # TODO: remove this method after tests
    #     self._data = data

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        """_draw_impl"""
        channels = bitmap.shape[2] if len(bitmap.shape) == 3 else 1
        non_zero_values = self.data > 0
        if channels >= 3:
            if isinstance(color, int):
                color = [color] * 3
            temp_mask = np.zeros(self.data.shape + (3,), dtype=np.uint8)
            temp_mask[non_zero_values] = color
            alpha = self.data / 255.0
            for i in range(3):
                canvas = self.to_bbox().get_cropped_numpy_slice(bitmap)[:, :, i]
                canvas[non_zero_values] = (canvas * (1 - alpha) + alpha * temp_mask[:, :, i])[
                    non_zero_values
                ]
        elif channels == 1:
            if isinstance(color, list):
                raise ValueError("Requires 'int' color for 1-channel image.")
            temp_mask = np.zeros(self.data.shape, dtype=np.uint8)
            temp_mask[non_zero_values] = color
            alpha_channel = np.zeros(self.data.shape, dtype=np.uint8)
            alpha_channel[non_zero_values] = self.data[non_zero_values]
            alpha = alpha_channel / 255.0
            self.to_bbox().get_cropped_numpy_slice(bitmap)[non_zero_values] = (
                1 - alpha[non_zero_values]
            ) * self.to_bbox().get_cropped_numpy_slice(bitmap)[non_zero_values] + alpha[
                non_zero_values
            ] * temp_mask[
                non_zero_values
            ]
        ###
        # temp_mask = np.zeros(self.data.shape + (3,), dtype=np.uint8)
        # alpha_channel = np.zeros(self.data.shape + (3,), dtype=np.uint8)

        # temp_mask[non_zero_values] = color
        # alpha_channel[..., -1][non_zero_values] = self.data[non_zero_values]
        # alpha = alpha_channel / 255.0

        # for i in range(dims):
        #     self.to_bbox().get_cropped_numpy_slice(bitmap)[..., i][non_zero_values] = (
        #         1 - alpha
        #     ) * self.to_bbox().get_cropped_numpy_slice(bitmap)[..., i][non_zero_values] + alpha * temp_mask[..., i][non_zero_values]
        ###

        # temp_mask = np.zeros(bitmap.shape, dtype=np.uint8)
        # alpha_channel = np.zeros(bitmap.shape, dtype=np.uint8)

        #######

        # # if len(self.data.shape) != 2 or bitmap.shape[2] < 4:
        # #     raise ValueError("Requires 4-channel image as input for drawing.")
        # # h, w = [:2]

        # # try100500
        # self.to_bbox().get_cropped_numpy_slice(temp_mask)[:, :, dims][self.data] = color
        # self.to_bbox().get_cropped_numpy_slice(alpha_channel)[:, :, -1][self.data] = self.data
        # alpha = self.to_bbox().get_cropped_numpy_slice(alpha_channel)[:, :, -1][self.data] / 255.0

        # self.to_bbox().get_cropped_numpy_slice(bitmap)[:, :, dims][
        #     self.data
        # ] = self.to_bbox().get_cropped_numpy_slice(bitmap)[:, :, dims][self.data] * (1 - alpha) + color * alpha
        #######

        # self.to_bbox().get_cropped_numpy_slice(temp_mask)[:, :, 3][non_zero_values] = color
        # self.to_bbox().get_cropped_numpy_slice(alpha_channel)[non_zero_values] = self.data[
        #     non_zero_values
        # ]
        # # alpha_channel = np.zeros((self.data.shape), dtype=np.uint8)
        # # alpha_channel = self.data[non_zero_values]
        # alpha = alpha_channel / 255.0

        # if dims == 1:
        #     bitmap[:, :, 0] = (1 - alpha) * bitmap[:, :, 0] + alpha * temp_mask[:, :, 0]
        #     # bitmap[:, :] = (1 - alpha[:, :, 0]) * bitmap[:, :] + alpha[:, :, 0] * temp_mask[
        #     #     :, :, 0
        #     # ]  # IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
        #     # bitmap[:, :] = (1 - alpha[:, :]) * bitmap[:, :] + alpha[:, :] * temp_mask[
        #     #     :, :
        #     # ] # IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
        #     bitmap[:, :] = (1 - alpha) * bitmap[:, :] + alpha * temp_mask[:, :]

        # elif dims >= 3:
        #     for i in range(dims):
        #         bitmap[:, :, i] = (1 - alpha[:, :, 0]) * bitmap[:, :, i] + alpha[:, :, 0] * temp_mask[
        #             :, :, i
        #         ]

    def _draw_contour_impl(self, alpha_mask, color, thickness=1, config=None):
        """_draw_contour_impl"""
        # pylint: disable=(no-member, unpacking-non-sequence)
        if StrictVersion(cv2.__version__) >= StrictVersion("4.0.0"):  # pylint: disable=no-member
            contours, _ = cv2.findContours(
                self.data.astype(np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        else:
            _, contours, _ = cv2.findContours(
                self.data.astype(np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        if contours is not None:
            for cont in contours:
                cont[:, :] += (
                    self.origin.col,
                    self.origin.row,
                )  # cont with shape (rows, ?, 2)
            cv2.drawContours(
                alpha_mask, contours, -1, color, thickness=thickness
            )
        # pylint: enable=(no-member, unpacking-non-sequence)

    @property
    def area(self) -> float:
        """
        AlphaMask area.

        :return: Area of current AlphaMask
        :rtype: :class:`float`
        :Usage example:

         .. code-block:: python

            print(figure.area)
            # Output: 88101.0
        """
        return float(np.count_nonzero(self._data))

    @staticmethod
    def base64_2_data(s: str) -> np.ndarray:
        """
        Convert base64 encoded string to numpy array. Supports both compressed and uncompressed masks.

        :param s: Input base64 encoded string.
        :type s: str
        :return: numpy array
        :rtype: :class:`np.ndarray`
        :Usage example:

         .. code-block:: python

              import supervisely as sly

              encoded_string = 'eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6'
              figure_data = sly.AlphaMask.base64_2_data(encoded_string)
              print(figure_data)

              uncompressed_string = 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA'
              mask = sly.AlphaMask.base64_2_data(uncompressed_string)
              print(mask)
        """
        try:
            z = zlib.decompress(base64.b64decode(s))
        except zlib.error:
            # If the string is not compressed, we'll not use zlib.
            img = Image.open(io.BytesIO(base64.b64decode(s)))
            return np.array(img)
        n = np.frombuffer(z, np.uint8)

        imdecoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)  # pylint: disable=no-member
        if (len(imdecoded.shape) == 3) and (imdecoded.shape[2] == 4):
            mask = imdecoded[:, :, 3]
        elif len(imdecoded.shape) == 2:
            mask = imdecoded
        else:
            raise RuntimeError("Wrong internal mask format.")
        return mask

    @staticmethod
    def data_2_base64(mask: np.ndarray) -> str:
        """
        Convert numpy array to base64 encoded string.

        :param mask: Bool numpy array.
        :type mask: np.ndarray
        :return: Base64 encoded string
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get annotation from API
            meta_json = api.project.get_meta(PROJECT_ID)
            meta = sly.ProjectMeta.from_json(meta_json)
            ann_info = api.annotation.download(IMAGE_ID)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)

            # Get AlphaMask from annotation
            for label in ann.labels:
                if type(label.geometry) == sly.AlphaMask:
                    figure = label.geometry

                    encoded_string = sly.AlphaMask.data_2_base64(figure.data)
                    print(encoded_string)
            # 'eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6'
        """
        img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
        # img_pil.putpalette([0, 0, 0, 255, 255, 255])
        bytes_io = io.BytesIO()
        img_pil.save(bytes_io, format="PNG", optimize=0)  # ?, transparency=0)
        bytes_enc = bytes_io.getvalue()
        return base64.b64encode(bytes_enc).decode("utf-8")
        # return base64.b64encode(zlib.compress(bytes_enc)).decode("utf-8")

    def skeletonize(self, method_id: SkeletonizeMethod) -> AlphaMask:
        """
        Compute the skeleton, medial axis transform or morphological thinning of AlphaMask.

        :param method_id: Method to convert bool numpy array.
        :type method_id: SkeletonizeMethod
        :return: AlphaMask object
        :rtype: :class:`AlphaMask<AlphaMask>`
        :Usage example:

         .. code-block:: python

            # Remember that AlphaMask class object is immutable, and we need to assign new instance of AlphaMask to a new variable
            skeleton_figure = figure.skeletonize(SkeletonizeMethod.SKELETONIZE)
            med_ax_figure = figure.skeletonize(SkeletonizeMethod.MEDIAL_AXIS)
            thin_figure = figure.skeletonize(SkeletonizeMethod.THINNING)
        """
        from skimage import morphology as skimage_morphology

        if method_id == SkeletonizeMethod.SKELETONIZE:
            method = skimage_morphology.skeletonize
        elif method_id == SkeletonizeMethod.MEDIAL_AXIS:
            method = skimage_morphology.medial_axis
        elif method_id == SkeletonizeMethod.THINNING:
            method = skimage_morphology.thin
        else:
            raise NotImplementedError("Method {!r} does't exist.".format(method_id))

        mask_u8 = self.data.astype(np.uint8)
        res_mask = method(mask_u8)
        return AlphaMask(data=res_mask, origin=self.origin)

    def to_contours(self) -> List[Polygon]:
        """
        Get list of contours in AlphaMask.

        :return: List of Polygon objects
        :rtype: :class:`List[Polygon]<supervisely.geometry.polygon.Polygon>`
        :Usage example:

         .. code-block:: python

            figure_contours = figure.to_contours()
        """
        origin, mask = self.origin, self.data
        if StrictVersion(cv2.__version__) >= StrictVersion("4.0.0"):  # pylint: disable=no-member
            contours, hier = cv2.findContours(  # pylint: disable=no-member
                mask.astype(np.uint8),
                mode=cv2.RETR_CCOMP,  # two-level hierarchy, to get polygons with holes # pylint: disable=no-member
                method=cv2.CHAIN_APPROX_SIMPLE,  # pylint: disable=no-member
            )
        else:
            _, contours, hier = cv2.findContours(  # pylint: disable=no-member
                mask.astype(np.uint8),
                mode=cv2.RETR_CCOMP,  # two-level hierarchy, to get polygons with holes # pylint: disable=no-member
                method=cv2.CHAIN_APPROX_SIMPLE,  # pylint: disable=no-member
            )
        if (hier is None) or (contours is None):
            return []

        res = []
        for idx, hier_pos in enumerate(hier[0]):
            next_idx, prev_idx, child_idx, parent_idx = hier_pos
            if parent_idx < 0:
                external = contours[idx][:, 0].tolist()
                internals = []
                while child_idx >= 0:
                    internals.append(contours[child_idx][:, 0])
                    child_idx = hier[0][child_idx][0]
                if len(external) > 2:
                    new_poly = Polygon(
                        exterior=row_col_list_to_points(external, flip_row_col_order=True),
                        interior=[
                            row_col_list_to_points(x, flip_row_col_order=True) for x in internals
                        ],
                    )
                    res.append(new_poly)

        offset_row, offset_col = origin.row, origin.col
        res = [x.translate(offset_row, offset_col) for x in res]
        return res

    def bitwise_mask(self, full_target_mask: np.ndarray, bit_op) -> AlphaMask:
        """
        Make bitwise operations between a given numpy array and AlphaMask.

        :param full_target_mask: Input numpy array.
        :type full_target_mask: np.ndarray
        :param bit_op: Type of bitwise operation(and, or, not, xor), uses `numpy logic <https://numpy.org/doc/stable/reference/routines.logic.html>`_ functions.
        :type bit_op: `Numpy logical operation <https://numpy.org/doc/stable/reference/routines.logic.html#logical-operations>`_
        :return: AlphaMask object or empty list
        :rtype: :class:`AlphaMask<AlphaMask>` or :class:`list`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            mask = np.array([[0, 0, 0, 0, 0],
                            [0, 50, 50, 50, 0],
                            [0, 50, 0, 50, 0],
                            [0, 50, 50, 50, 0],
                            [0, 0, 0, 0, 0]], dtype=np.uint8)

            figure = sly.AlphaMask(mask)

            array = np.array([[0, 0, 0, 0, 0],
                             [0, 50, 50, 50, 0],
                             [0, 0, 50, 0, 0],
                             [0, 0, 50, 0, 0],
                             [0, 0, 0, 0, 0]], dtype=np.uint8)

            bitwise_figure = figure.bitwise_mask(array, np.logical_and)
            print(bitwise_figure.data)
        """
        full_size = full_target_mask.shape[:2]
        origin, mask = self.origin, self.data
        full_size_mask = np.full(full_size, 0, np.uint8)
        full_size_mask[
            origin.row : origin.row + mask.shape[0],
            origin.col : origin.col + mask.shape[50],
        ] = mask

        new_mask = bit_op(full_target_mask, full_size_mask).astype(np.uint8)
        if new_mask.sum() == 0:
            return []
        new_mask = new_mask[
            origin.row : origin.row + mask.shape[0],
            origin.col : origin.col + mask.shape[1],
        ]
        return AlphaMask(data=new_mask, origin=origin.clone())

    @classmethod
    def allowed_transforms(cls):
        """allowed_transforms"""
        from supervisely.geometry.any_geometry import AnyGeometry
        from supervisely.geometry.polygon import Polygon
        from supervisely.geometry.rectangle import Rectangle

        return [AnyGeometry, Polygon, Rectangle]

    @classmethod
    def from_path(cls, path: str) -> AlphaMask:
        """
        Read alpha_channel from image by path.

        :param path: Path to image
        :type path: str
        :return: AlphaMask
        :rtype: AlphaMask
        """
        img = read(path)
        return AlphaMask(img[:, :, 0])
