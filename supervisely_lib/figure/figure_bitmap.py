# coding: utf-8

import base64
import zlib
import io
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image

from .abstract_figure import AbstractFigure
from .rectangle import Rect


class FigureBitmap(AbstractFigure):
    def _set_bmp(self, origin, mask):
        self.data['bitmap'] = {
            'origin': origin,
            'np': mask
        }

    def _get_slice(self):
        bbox = self.get_bbox_int()
        slices = slice(bbox.top, bbox.bottom), slice(bbox.left, bbox.right)
        return slices

    @classmethod
    def _crop_and_compress(cls, origin, mask, crop_rect):
        origin, mask = FigureBitmap.crop_mask(origin, mask, crop_rect)
        if origin is None:
            return None, None

        origin, mask = FigureBitmap.compress_mask(mask, origin=origin)
        return origin, mask

    def get_origin_mask(self):
        bmp_dict = self.data['bitmap']
        return bmp_dict['origin'], bmp_dict['np']

    def normalize(self, img_size_wh):
        origin, mask = self.get_origin_mask()

        if (len(mask.shape) != 2) or (mask.dtype != np.bool):
            raise RuntimeError('Wrong mask type in bitmap (internal fmt)')
        if len(origin) != 2:
            raise RuntimeError('Wrong origin in bitmap (internal fmt)')

        if min(mask.shape) < 1:
            return []  # no data

        # @TODO: require int coords from @tony
        origin = [round(x) for x in origin]  # ok, round casts to int

        # crop to image bounds, remove unused margins; may produce empty result
        im_rect = Rect.from_size(img_size_wh)
        origin, mask = self._crop_and_compress(origin, mask, im_rect)
        if origin is None:
            return []
        else:
            self._set_bmp(origin, mask)
            return [self]

    def crop(self, rect):
        rect_rounded = rect.round()
        if rect_rounded.is_empty:
            return []

        origin, mask = self.get_origin_mask()
        origin, mask = self._crop_and_compress(origin, mask, rect_rounded)
        if origin is None:
            return []
        else:
            self._set_bmp(origin, mask)
            return [self]

    def rotate(self, rotator):
        full_img_mask = self.to_bool_mask(rotator.src_imsize_wh[::-1]).astype(np.uint8)
        new_mask = rotator.rotate_img(full_img_mask, use_inter_nearest=True).astype(bool)
        new_img_bbox = Rect.from_size(rotator.new_imsize_wh)
        origin, mask = self._crop_and_compress((0, 0), new_mask, new_img_bbox)
        self._set_bmp(origin, mask)

    def resize(self, resizer):
        full_img_mask = self.to_bool_mask(resizer.src_size_wh[::-1]).astype(np.uint8)
        new_mask = resizer.resize_img(full_img_mask, use_nearest=True).astype(bool)
        new_img_bbox = Rect.from_size((resizer.new_w, resizer.new_h))
        origin, mask = self._crop_and_compress((0, 0), new_mask, new_img_bbox)
        self._set_bmp(origin, mask)

    def shift(self, delta):
        origin, mask = self.get_origin_mask()
        origin = (origin[0] + delta[0], origin[1] + delta[1])
        self._set_bmp(origin, mask)

    def flip(self, is_horiz, img_wh):
        origin, mask = self.get_origin_mask()
        origin = list(origin)
        mask_h, mask_w = mask.shape
        if is_horiz:
            mask = mask[::-1, :]
            origin[1] = img_wh[1] - origin[1] - mask_h
        else:
            mask = mask[:, ::-1]
            origin[0] = img_wh[0] - origin[0] - mask_w
        self._set_bmp(origin, mask)

    # returns Rect with integer bounds
    def get_bbox_int(self):
        origin, mask = self.get_origin_mask()
        rect = Rect.from_arr(mask)
        rect = rect.move((origin[0], origin[1]))
        return rect

    def get_bbox(self):
        return self.get_bbox_int()

    def to_bool_mask(self, shape_hw):
        origin, mask = self.get_origin_mask()
        slices = self._get_slice()
        res_mask = np.zeros(shape_hw, np.bool)
        res_mask[slices][mask] = True
        return res_mask

    def draw(self, bitmap, color):
        origin, mask = self.get_origin_mask()
        slices = self._get_slice()
        bitmap[slices][mask] = color

    # m/b not fast
    def draw_contour(self, bitmap, color, thickness):
        origin, mask = self.get_origin_mask()
        _, contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contours is not None:
            origin_t = tuple(origin)
            for cont in contours:
                cont[:, :] += origin_t  # cont with shape (rows, ?, 2)
            cv2.drawContours(bitmap, contours, -1, color, thickness=thickness)

    def get_area(self):
        _, mask = self.get_origin_mask()
        res = np.sum(mask)
        return res

    def pack(self):
        origin, mask = self.get_origin_mask()
        res_origin = [int(x) for x in origin]
        res_mask_encoded = FigureBitmap.mask_2_base64(mask)
        packed = deepcopy(self.data)
        packed['bitmap'] = {
            'origin': res_origin,
            'data': res_mask_encoded
        }
        return packed

    @classmethod
    def from_packed(cls, packed_obj):
        obj = packed_obj
        origin = packed_obj['bitmap'].get('origin', (0, 0))
        mask = cls.base64_2_mask(packed_obj['bitmap']['data'])
        obj['bitmap'] = {
            'origin': origin,
            'np': mask
        }
        return cls(obj)

    # @TODO: rewrite, validate, generalize etc
    # returns iterable
    @classmethod
    def from_mask(cls, class_title, image_size_wh, origin, mask):
        new_data = {
            'bitmap': {
                'origin': origin,
                'np': mask,
            },
            'type': 'bitmap',
            'classTitle': class_title,
            'description': '',
            'tags': [],
            'points': {'interior': [], 'exterior': []},
        }
        temp = cls(new_data)
        res = temp.normalize(image_size_wh)
        return res

    @staticmethod
    def base64_2_mask(s):
        z = zlib.decompress(base64.b64decode(s))
        n = np.fromstring(z, np.uint8)

        imdecoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
        if (len(imdecoded.shape) == 3) and (imdecoded.shape[2] >= 4):
            mask = imdecoded[:, :, 3].astype(bool)  # 4-channel imgs
        elif len(imdecoded.shape) == 2:
            mask = imdecoded.astype(bool)  # flat 2d mask
        else:
            raise RuntimeError('Wrong internal mask format.')

        # out mask must be 2d bool array
        return mask

    @staticmethod
    def mask_2_base64(mask):
        img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
        img_pil.putpalette([0, 0, 0, 255, 255, 255])
        bytes_io = io.BytesIO()
        img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
        bytes_enc = bytes_io.getvalue()
        res = base64.b64encode(zlib.compress(bytes_enc)).decode('utf-8')
        return res

    # removes empty margins
    @staticmethod
    def compress_mask(mask, origin=None):
        if not np.any(mask):
            return None, None
        rows = np.any(mask, axis=1)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cols = np.any(mask, axis=0)
        cmin, cmax = np.where(cols)[0][[0, -1]]
        res_origin = (cmin, rmin)
        res_mask = mask[rmin:rmax+1, cmin:cmax+1].copy()
        if origin is not None:
            res_origin = [a + b for a, b in zip(origin, res_origin)]
        return res_origin, res_mask

    # expects integer origin & crop_rect
    @staticmethod
    def crop_mask(origin, mask, crop_rect):
        mask_rect = Rect.from_arr(mask)  # in coords of mask
        crop_rect_rel_to_mask = crop_rect.move((-origin[0], -origin[1]))  # in coords of mask
        inters_rect = mask_rect.intersection(crop_rect_rel_to_mask)  # in coords of mask
        if inters_rect.is_empty:
            return None, None
        left, top, right, bottom = inters_rect.data
        res_mask = mask[top:bottom, left:right]
        res_origin = (origin[0] + left, origin[1] + top)  # in coords of source img
        return res_origin, res_mask
