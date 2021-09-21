# coding: utf-8
import base64
from enum import Enum
import zlib
import io
import cv2
import numpy as np
from distutils.version import StrictVersion

from PIL import Image
from skimage import morphology as skimage_morphology

from supervisely_lib.geometry.bitmap_base import BitmapBase, resize_origin_and_bitmap
from supervisely_lib.geometry.point_location import PointLocation, row_col_list_to_points
from supervisely_lib.geometry.polygon import Polygon
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.geometry.constants import BITMAP


class SkeletonizeMethod(Enum):
    SKELETONIZE = 0
    MEDIAL_AXIS = 1
    THINNING = 2


def _find_mask_tight_bbox(raw_mask: np.ndarray) -> Rectangle:
    rows = list(np.any(raw_mask, axis=1).tolist())  # Redundant conversion to list to help PyCharm static analysis.
    cols = list(np.any(raw_mask, axis=0).tolist())
    top_margin = rows.index(True)
    bottom_margin = rows[::-1].index(True)
    left_margin = cols.index(True)
    right_margin = cols[::-1].index(True)
    return Rectangle(top=top_margin, left=left_margin, bottom=len(rows) - 1 - bottom_margin,
                     right=len(cols) - 1 - right_margin)


class Bitmap(BitmapBase):
    '''
    This is a class for creating and using Bitmaps for Labels.
    '''
    @staticmethod
    def geometry_name():
        return 'bitmap'

    def __init__(self, data: np.ndarray, origin: PointLocation = None,
                 sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        '''
        The constructor for Bitmap class.
        :param data: bool numpy array
        :param origin: points (x and y coordinates) of the top left corner of a bitmap, i.e. the position of the
        bitmap within the image
        '''

        if data.dtype != np.bool:
            raise ValueError('Bitmap mask data must be a boolean numpy array. Instead got {}.'.format(str(data.dtype)))

        # Call base constructor first to do the basic dimensionality checks.
        super().__init__(data=data, origin=origin, expected_data_dims=2,
                         sly_id=sly_id, class_id=class_id,
                         labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

        # Crop the empty margins of the boolean mask right away.
        if not np.any(data):
            raise ValueError('Creating a bitmap with an empty mask (no pixels set to True) is not supported.')
        data_tight_bbox = _find_mask_tight_bbox(self._data)
        self._origin = self._origin.translate(drow=data_tight_bbox.top, dcol=data_tight_bbox.left)
        self._data = data_tight_bbox.get_cropped_numpy_slice(self._data)

    @classmethod
    def _impl_json_class_name(cls):
        return BITMAP

    def rotate(self, rotator):
        '''
        The function rotate Bitmap mask with a given rotator
        (ImageRotator class object contain size of image and angle to rotate)
        :param rotator: ImageRotator class object
        :return: Bitmap class object
        '''
        full_img_mask = np.zeros(rotator.src_imsize, dtype=np.uint8)
        self.draw(full_img_mask, 1)
        # TODO this may break for one-pixel masks (it can disappear during rotation). Instead, rotate every pixel
        #  individually and set it in the resulting bitmap.
        new_mask = rotator.rotate_img(full_img_mask, use_inter_nearest=True).astype(np.bool)
        return Bitmap(data=new_mask)

    def crop(self, rect):
        '''
        Crop the current Bitmap object with a given rectangle
        :param rect: Rectangle class object
        :return: Bitmap class object
        '''
        maybe_cropped_bbox = self.to_bbox().crop(rect)
        if len(maybe_cropped_bbox) == 0:
            return []
        else:
            [cropped_bbox] = maybe_cropped_bbox
            cropped_bbox_relative = cropped_bbox.translate(drow=-self.origin.row, dcol=-self.origin.col)
            cropped_mask = cropped_bbox_relative.get_cropped_numpy_slice(self._data)
            if not np.any(cropped_mask):
                return []
            return [Bitmap(data=cropped_mask, origin=PointLocation(row=cropped_bbox.top, col=cropped_bbox.left))]

    def resize(self, in_size, out_size):
        '''
        Resize the current Bitmap to match a certain size
        :param in_size: input image size
        :param out_size: output image size
        :return: Bitmap class object
        '''
        scaled_origin, scaled_data = resize_origin_and_bitmap(self._origin, self._data.astype(np.uint8), in_size,
                                                              out_size)
        # TODO this might break if a sparse mask is resized too thinly. Instead, resize every pixel individually and set
        #  it in the resulting bitmap.
        return Bitmap(data=scaled_data.astype(np.bool), origin=scaled_origin)

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        self.to_bbox().get_cropped_numpy_slice(bitmap)[self.data] = color

    def _draw_contour_impl(self, bitmap, color, thickness=1, config=None):
        if StrictVersion(cv2.__version__) >= StrictVersion('4.0.0'):
            contours, _ = cv2.findContours(self.data.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, _ = cv2.findContours(self.data.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contours is not None:
            for cont in contours:
                cont[:, :] += (self.origin.col, self.origin.row)  # cont with shape (rows, ?, 2)
            cv2.drawContours(bitmap, contours, -1, color, thickness=thickness)

    @property
    def area(self):
        '''
        :return: area of current Bitmap
        '''
        return float(self._data.sum())

    @staticmethod
    def base64_2_data(s: str) -> np.ndarray:
        '''
        The function base64_2_data convert base64 encoded string to numpy
        :param s: string
        :return: numpy array
        '''
        z = zlib.decompress(base64.b64decode(s))
        n = np.frombuffer(z, np.uint8)

        imdecoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
        if (len(imdecoded.shape) == 3) and (imdecoded.shape[2] >= 4):
            mask = imdecoded[:, :, 3].astype(bool)  # 4-channel imgs
        elif len(imdecoded.shape) == 2:
            mask = imdecoded.astype(bool)  # flat 2d mask
        else:
            raise RuntimeError('Wrong internal mask format.')
        return mask

    @staticmethod
    def data_2_base64(mask: np.ndarray) -> str:
        '''
        The function data_2_base64 convert numpy array to base64 encoded string
        :param mask: numpy array
        :return: string
        '''
        img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
        img_pil.putpalette([0, 0, 0, 255, 255, 255])
        bytes_io = io.BytesIO()
        img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
        bytes_enc = bytes_io.getvalue()
        return base64.b64encode(zlib.compress(bytes_enc)).decode('utf-8')

    def skeletonize(self, method_id: SkeletonizeMethod):
        '''
        The function compute the skeleton, medial axis transform or morphological thinning of Bitmap
        :param method_id: method to convert binary image(numpy array)
        :return: Bitmap class object
        '''
        if method_id == SkeletonizeMethod.SKELETONIZE:
            method = skimage_morphology.skeletonize
        elif method_id == SkeletonizeMethod.MEDIAL_AXIS:
            method = skimage_morphology.medial_axis
        elif method_id == SkeletonizeMethod.THINNING:
            method = skimage_morphology.thin
        else:
            raise NotImplementedError('Method {!r} does\'t exist.'.format(method_id))

        mask_u8 = self.data.astype(np.uint8)
        res_mask = method(mask_u8).astype(bool)
        return Bitmap(data=res_mask, origin=self.origin)

    def to_contours(self):
        '''
        The function to_contours get list of contours(in Polygon class objects) of all objects in Bitmap
        :return: list of contours
        '''

        origin, mask = self.origin, self.data
        if StrictVersion(cv2.__version__) >= StrictVersion('4.0.0'):
            contours, hier = cv2.findContours(
                mask.astype(np.uint8),
                mode=cv2.RETR_CCOMP,  # two-level hierarchy, to get polygons with holes
                method=cv2.CHAIN_APPROX_SIMPLE
            )
        else:
            _, contours, hier = cv2.findContours(
                mask.astype(np.uint8),
                mode=cv2.RETR_CCOMP,  # two-level hierarchy, to get polygons with holes
                method=cv2.CHAIN_APPROX_SIMPLE
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
                    new_poly = Polygon(exterior=row_col_list_to_points(external, flip_row_col_order=True),
                                       interior=[row_col_list_to_points(x, flip_row_col_order=True) for x in internals])
                    res.append(new_poly)

        offset_row, offset_col = origin.row, origin.col
        res = [x.translate(offset_row, offset_col) for x in res]
        return res

    def bitwise_mask(self, full_target_mask: np.ndarray, bit_op):
        '''
        The function bitwise_mask performs bitwise operations between a given numpy array and Bitmap object data
        :param full_target_mask: numpy array
        :param bit_op: type of bitwise operation(or, and, xor)
        :return: Bitmap class object or empty list
        '''
        full_size = full_target_mask.shape[:2]
        origin, mask = self.origin, self.data
        full_size_mask = np.full(full_size, False, bool)
        full_size_mask[origin.row:origin.row + mask.shape[0], origin.col:origin.col + mask.shape[1]] = mask

        new_mask = bit_op(full_target_mask, full_size_mask).astype(bool)
        if new_mask.sum() == 0:
            return []
        new_mask = new_mask[origin.row:origin.row + mask.shape[0], origin.col:origin.col + mask.shape[1]]
        return Bitmap(data=new_mask, origin=origin.clone())

    @classmethod
    def allowed_transforms(cls):
        from supervisely_lib.geometry.any_geometry import AnyGeometry
        from supervisely_lib.geometry.polygon import Polygon
        from supervisely_lib.geometry.rectangle import Rectangle
        return [AnyGeometry, Polygon, Rectangle]
