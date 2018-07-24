# coding: utf-8

import os
import os.path as osp
import base64
import functools

import cv2
import numpy as np
import skimage.transform
from PIL import Image


# dsize as WxH
# designed to replace cv2.resize INTER_NEAREST
# works with different dtypes and number of channels; speed isn't guaranteed
def resize_inter_nearest(src_img, dsize=None, fx=0, fy=0):
    if dsize is not None:
        target_shape = (dsize[1], dsize[0])
    else:
        target_shape = (np.round(fy * src_img.shape[0]), np.round(fx * src_img.shape[1]))

    if target_shape[0] <= 0 or target_shape[1] <= 0:
        raise RuntimeError('Wrong resize parameters.')

    res = skimage.transform.resize(
        src_img, target_shape, order=0, preserve_range=True, mode='constant').astype(src_img.dtype)
    return res


class ImgProto:
    @classmethod
    def img2str(cls, img):
        encoded = cv2.imencode('.png', img)[1].tostring()
        return base64.b64encode(encoded).decode('utf-8')

    # np.uint8, original shape
    @classmethod
    def str2img_original(cls, s):
        n = np.fromstring(base64.b64decode(s), np.uint8)
        imdecoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
        res = imdecoded.astype(np.uint8)
        return res

    # np.uint8, alpha channel from color image or source single channel
    @classmethod
    def str2img_single_channel(cls, s):
        imdecoded = cls.str2img_original(s)
        if (len(imdecoded.shape) == 3) and (imdecoded.shape[2] >= 4):
            mask = imdecoded[:, :, 3]  # 4-channel imgs
        elif len(imdecoded.shape) == 2:
            mask = imdecoded  # flat 2d mask
        else:
            raise RuntimeError('Wrong internal mask format.')
        return mask


# with np.uint8; for visualization; considers black fg as transparent
def overlay_images(bkg_img, fg_img, fg_coeff):
    comb_img = (fg_coeff * fg_img + (1 - fg_coeff) * bkg_img).astype(np.uint8)

    black_mask = (fg_img[:, :, 0] == 0) & (fg_img[:, :, 1] == 0) & (fg_img[:, :, 2] == 0)
    comb_img[black_mask] = bkg_img[black_mask]
    comb_img = np.clip(comb_img, 0, 255)

    return comb_img


class ImportImgLister:
    _included_extensions = ['jpg', 'jpeg', 'bmp', 'png', ]
    extensions = ['.' + x for x in _included_extensions + [x.upper() for x in _included_extensions]]

    @classmethod
    def list_images(cls, dir_):
        fnames = (f.name for f in os.scandir(dir_) if f.is_file())
        img_names = list(filter(lambda x: osp.splitext(x)[1] in cls.extensions, fnames))
        return img_names


def image_transpose_exif(im):
    """
        Apply Image.transpose to ensure 0th row of pixels is at the visual
        top of the image, and 0th column is the visual left-hand side.
        Return the original image if unable to determine the orientation.

        As per CIPA DC-008-2012, the orientation field contains an integer,
        1 through 8. Other values are reserved.
    """

    exif_orientation_tag = 0x0112
    exif_transpose_sequences = [                   # Val  0th row  0th col
        [],                                        #  0    (reserved)
        [],                                        #  1   top      left
        [Image.FLIP_LEFT_RIGHT],                   #  2   top      right
        [Image.ROTATE_180],                        #  3   bottom   right
        [Image.FLIP_TOP_BOTTOM],                   #  4   bottom   left
        [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],  #  5   left     top
        [Image.ROTATE_270],                        #  6   right    top
        [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],  #  7   right    bottom
        [Image.ROTATE_90],                         #  8   left     bottom
    ]

    seq = exif_transpose_sequences[im._getexif()[exif_orientation_tag]]
    return functools.reduce(type(im).transpose, seq, im)
