# coding: utf-8

# docs
from __future__ import annotations

import base64
import io
import zlib
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from supervisely.geometry.bitmap import Bitmap, _find_mask_tight_bbox
from supervisely.geometry.bitmap_base import BitmapBase
from supervisely.geometry.point_location import PointLocation
from supervisely.imaging.image import read



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

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        """_draw_impl"""
        channels = bitmap.shape[2] if len(bitmap.shape) == 3 else 1
        non_zero_values = self.data > 0
        alpha = self.data / 255.0
        if channels >= 3:
            if isinstance(color, int):
                color = [color] * 3
            temp_mask = np.zeros(self.data.shape + (3,), dtype=np.uint8)
            temp_mask[non_zero_values] = color
            for i in range(3):
                canvas = self.to_bbox().get_cropped_numpy_slice(bitmap)[:, :, i]
                canvas[non_zero_values] = (canvas * (1 - alpha) + alpha * temp_mask[:, :, i])[
                    non_zero_values
                ]
        elif channels == 1:
            temp_mask = np.zeros(self.data.shape, dtype=np.uint8)
            temp_mask[non_zero_values] = color
            canvas = self.to_bbox().get_cropped_numpy_slice(bitmap)
            canvas[non_zero_values] = (canvas * (1 - alpha) + alpha * temp_mask)[non_zero_values]


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
            mask = imdecoded[:, :, 3] # pylint: disable=unsubscriptable-object
        elif len(imdecoded.shape) == 2:
            mask = imdecoded
        else:
            raise RuntimeError("Wrong internal mask format.")
        return mask

    @classmethod
    def allowed_transforms(cls):
        """allowed_transforms"""
        from supervisely.geometry.any_geometry import AnyGeometry
        from supervisely.geometry.polygon import Polygon
        from supervisely.geometry.rectangle import Rectangle
        from supervisely.geometry.bitmap import Bitmap

        return [AnyGeometry, Bitmap, Polygon, Rectangle]

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
