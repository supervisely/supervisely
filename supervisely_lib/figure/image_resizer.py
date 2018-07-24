# coding: utf-8

import cv2
import numpy as np

from ..utils.imaging import resize_inter_nearest


class ImageResizer:
    def _determine_resize_params(self, src_size_wh, res_size_wh, keep):
        w, h = src_size_wh
        res_width, res_height = res_size_wh
        if keep:
            if res_height == -1:
                scale = res_width / w
                new_h = int(round(h * scale))
                new_w = res_width
            elif res_width == -1:
                scale = res_height / h
                new_h = res_height
                new_w = int(round(w * scale))
            else:
                scale = min(res_height / h, res_width / w)
                new_h = res_height
                new_w = res_width

            self._resize_args = [None]
            self._resize_kwargs = {'fx': scale, 'fy': scale}
            self._scale_x, self._scale_y = scale, scale
            left, top = (new_w - np.round(scale * w).astype('int')) // 2, (new_h - np.round(scale * h).astype('int')) // 2
        else:
            self._scale_x, self._scale_y = res_width / w, res_height / h
            self._resize_args = [(res_width, res_height)]
            self._resize_kwargs = {}
            new_h, new_w = res_height, res_width
            left, top = 0, 0
        self.new_h, self.new_w = new_h, new_w
        self._left, self._top = left, top

    def __init__(self, src_size_wh, res_size_wh, keep=False):
        self.src_size_wh = tuple(src_size_wh)
        self._determine_resize_params(src_size_wh, res_size_wh, keep)

    def _get_res_image_shape(self, img):
        if len(img.shape) > 2:
            res_shape = (self.new_h, self.new_w, img.shape[2])
        else:
            res_shape = (self.new_h, self.new_w)
        return res_shape

    def resize_img(self, img, use_nearest=False):
        if use_nearest:
            inter_img = resize_inter_nearest(img, *self._resize_args, **self._resize_kwargs)
        else:
            inter_img = cv2.resize(img, *self._resize_args, **self._resize_kwargs).astype(img.dtype)
        res_img = np.zeros(self._get_res_image_shape(inter_img), img.dtype)
        res_img[self._top:self._top + inter_img.shape[0], self._left:self._left + inter_img.shape[1]] = inter_img
        return res_img

    def transform_coords(self, x):
        x[:, 0] = x[:, 0] * self._scale_x + self._left
        x[:, 1] = x[:, 1] * self._scale_y + self._top

