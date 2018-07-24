# coding: utf-8

import sys
import collections
import math

import cv2
import numpy as np
from shapely.geometry import Polygon

from .rectangle import Rect


# works when called from class- or instance methods
def not_implemeted_method():
    level_up = sys._getframe(1)
    top_func_name = level_up.f_code.co_name
    top_self = level_up.f_locals.get('self', None)
    if top_self is not None:
        top_class = top_self.__class__
    else:
        top_class = level_up.f_locals['cls']
    raise NotImplementedError("Method {} should be implemented in {}.".format(top_func_name, top_class))


# image will contain rect; saves center point; returns non-neg delta
def expand_image_with_rect(img, req_rect):
    src_im_rect = Rect.from_arr(img)
    if src_im_rect.contains(req_rect):
        return img, (0, 0)

    src_ct = [int(x + .5) for x in src_im_rect.center]
    new_w2 = max(src_ct[0] - req_rect.left, req_rect.right - src_ct[0])
    new_h2 = max(src_ct[1] - req_rect.top, req_rect.bottom - src_ct[1])
    exp_w = max(src_im_rect.width, 2 * new_w2)
    exp_h = max(src_im_rect.height, 2 * new_h2)
    delta_x = (exp_w - src_im_rect.width) // 2
    delta_y = (exp_h - src_im_rect.height) // 2

    exp_img = np.zeros((exp_h, exp_w, img.shape[2]), dtype=img.dtype)
    exp_img[delta_y:delta_y+src_im_rect.height, delta_x:delta_x+src_im_rect.width] = img

    return exp_img, (delta_x, delta_y)


# requires correct rect (non-emoty, integer, in img bounds)
def crop_image_with_rect(img, rect_to_crop):
    rtc = rect_to_crop
    img = img[rtc.top:rtc.bottom, rtc.left:rtc.right, :]
    return img
