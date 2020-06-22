# coding: utf-8
import numpy as np

from supervisely_lib.geometry.constants import DATA, ORIGIN, GEOMETRY_SHAPE, GEOMETRY_TYPE, \
                                               LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID, CLASS_ID
from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.point_location import PointLocation
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.imaging.image import resize_inter_nearest, restore_proportional_size


# TODO: rename to resize_bitmap_and_origin
def resize_origin_and_bitmap(origin: PointLocation, bitmap: np.ndarray, in_size, out_size):
    '''
    Change PointLocation and resize numpy array to match a certain size
    :return: new PointLocation class object and numpy array
    '''
    new_size = restore_proportional_size(in_size=in_size, out_size=out_size)

    row_scale = new_size[0] / in_size[0]
    col_scale = new_size[1] / in_size[1]

    # TODO: Double check (+restore_proportional_size) or not? bitmap.shape and in_size are equal?
    # Make sure the resulting size has at least one pixel in every direction (i.e. limit the shrinkage to avoid having
    # empty bitmaps as a result).
    scaled_rows = max(round(bitmap.shape[0] * row_scale), 1)
    scaled_cols = max(round(bitmap.shape[1] * col_scale), 1)

    scaled_origin = PointLocation(row=round(origin.row * row_scale), col=round(origin.col * col_scale))
    scaled_bitmap = resize_inter_nearest(bitmap, (scaled_rows, scaled_cols))
    return scaled_origin, scaled_bitmap


class BitmapBase(Geometry):
    def __init__(self, data: np.ndarray, origin: PointLocation = None, expected_data_dims=None,
                 sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        super().__init__(sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)
        """
        :param origin: PointLocation class object
        :param data: np.ndarray
        """
        if origin is None:
            origin = PointLocation(row=0, col=0)

        if not isinstance(origin, PointLocation):
            raise TypeError('BitmapBase "origin" argument must be "PointLocation" object!')

        if not isinstance(data, np.ndarray):
            raise TypeError('BitmapBase "data" argument must be numpy array object!')

        data_dims = len(data.shape)
        if expected_data_dims is not None and data_dims != expected_data_dims:
            raise ValueError('BitmapBase "data" argument must be a {}-dimensional numpy array. '
                             'Instead got {} dimensions'.format(expected_data_dims, data_dims))

        self._origin = origin.clone()
        self._data = data.copy()

    @classmethod
    def _impl_json_class_name(cls):
        """Descendants must implement this to return key string to look up serialized representation in a JSON dict."""
        raise NotImplementedError()

    @staticmethod
    def base64_2_data(s: str) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def data_2_base64(data: np.ndarray) -> str:
        raise NotImplementedError()

    def to_json(self):
        '''
        The function to_json convert bitmap to json format
        :return: Bitmap in json format(dict)
        '''
        res = {
            self._impl_json_class_name(): {
                ORIGIN: [self.origin.col, self.origin.row],
                DATA: self.data_2_base64(self.data)
            },
            GEOMETRY_SHAPE: self.geometry_name(),
            GEOMETRY_TYPE: self.geometry_name(),
        }
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, json_data):
        '''
        The function from_json convert bitmap from json format to Bitmap class object.
        :param json_data: input bitmap in json format
        :return: Bitmap class object
        '''
        json_root_key = cls._impl_json_class_name()
        if json_root_key not in json_data:
            raise ValueError(
                'Data must contain {} field to create MultichannelBitmap object.'.format(json_root_key))

        if ORIGIN not in json_data[json_root_key] or DATA not in json_data[json_root_key]:
            raise ValueError('{} field must contain {} and {} fields to create MultichannelBitmap object.'.format(
                json_root_key, ORIGIN, DATA))

        col, row = json_data[json_root_key][ORIGIN]
        data = cls.base64_2_data(json_data[json_root_key][DATA])

        labeler_login = json_data.get(LABELER_LOGIN, None)
        updated_at = json_data.get(UPDATED_AT, None)
        created_at = json_data.get(CREATED_AT, None)
        sly_id = json_data.get(ID, None)
        class_id = json_data.get(CLASS_ID, None)
        return cls(data=data, origin=PointLocation(row=row, col=col),
                   sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

    @property
    def origin(self) -> PointLocation:
        '''
        The function origin return copy of Bitmap class object PointLocation
        :return: PointLocation class object
        '''
        return self._origin.clone()

    @property
    def data(self) -> np.ndarray:
        '''
        The function data return copy of Bitmap class object data(numpy array)
        :return: numpy array
        '''
        return self._data.copy()

    def translate(self, drow, dcol):
        '''
        The function translate shifts the object by a certain number of pixels and return the copy of the current Bitmap object
        :param drow: horizontal shift
        :param dcol: vertical shift
        :return: Bitmap class object with new PointLocation
        '''
        translated_origin = self.origin.translate(drow, dcol)
        return self.__class__(data=self.data, origin=translated_origin)

    def fliplr(self, img_size):
        '''
        The function fliplr flip the current Bitmap object in horizontal and return the copy of the
        current Bitmap object
        :param img_size: size of the image
        :return: Bitmap class object with new data(numpy array) and PointLocation
        '''
        flipped_mask = np.flip(self.data, axis=1)
        flipped_origin = PointLocation(row=self.origin.row, col=(img_size[1] - flipped_mask.shape[1] - self.origin.col))
        return self.__class__(data=flipped_mask, origin=flipped_origin)

    def flipud(self, img_size):
        '''
        The function fliplr flip the current Bitmap object in vertical and return the copy of the
        current Bitmap object
        :param img_size: size of the image
        :return: Bitmap class object with new data(numpy array) and PointLocation
        '''
        flipped_mask = np.flip(self.data, axis=0)
        flipped_origin = PointLocation(row=(img_size[0] - flipped_mask.shape[0] - self.origin.row), col=self.origin.col)
        return self.__class__(data=flipped_mask, origin=flipped_origin)

    def scale(self, factor):
        '''
        The function scale change scale of the current Bitmap object with a given factor
        :param factor: float scale parameter
        :return: Bitmap class object with new data(numpy array) and PointLocation
        '''
        new_rows = round(self._data.shape[0] * factor)
        new_cols = round(self._data.shape[1] * factor)
        mask = self._resize_mask(self.data, new_rows, new_cols)
        origin = self.origin.scale(factor)
        return self.__class__(data=mask, origin=origin)

    @staticmethod
    def _resize_mask(mask, out_rows, out_cols):
        return resize_inter_nearest(mask.astype(np.uint8), (out_rows, out_cols)).astype(np.bool)

    def to_bbox(self):
        '''
        The function to_bbox create Rectangle class object from current Bitmap class object
        :return: Rectangle class object
        '''
        return Rectangle.from_array(self._data).translate(drow=self._origin.row, dcol=self._origin.col)
