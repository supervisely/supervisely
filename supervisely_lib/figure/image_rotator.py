# coding: utf-8

import math

import cv2
import numpy as np

from .rectangle import Rect


# to rotate image & objects wrt source img center
# output image will contain all 'pixels' from source img
class ImageRotator:
    # to get rect with max 'coloured' area in rotated img
    def _calc_inner_crop(self):
        a_ccw = np.deg2rad(self.angle_degrees_ccw)
        quadrant = int(math.floor(a_ccw / (math.pi / 2))) & 3
        sign_alpha = a_ccw if ((quadrant & 1) == 0) else math.pi - a_ccw
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        w, h = self.src_imsize_wh
        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        largest_w, largest_h = bb_w - 2 * x, bb_h - 2 * y

        new_w, new_h = self.new_imsize_wh
        left = int((new_w - largest_w) * 0.5)
        right = int((new_w + largest_w) * 0.5)
        top = int((new_h - largest_h) * 0.5)
        bottom = int((new_h + largest_h) * 0.5)
        some_inner_crop = Rect(left, top, right, bottom)
        new_img_bbox = Rect.from_size(self.new_imsize_wh)
        self.inner_crop = new_img_bbox.intersection(some_inner_crop)  # to ensure

    def __init__(self, imsize_wh, angle_degrees_ccw):
        self.src_imsize_wh = tuple(imsize_wh)
        self.angle_degrees_ccw = angle_degrees_ccw

        def wh_to_center(wh):
            return wh[0] / 2.0, wh[1] / 2.0    # float xy

        self.img_center = wh_to_center(self.src_imsize_wh)

        rot_mat = np.vstack([
            cv2.getRotationMatrix2D(self.img_center, self.angle_degrees_ccw, 1.0),
            [0, 0, 1]
        ])
        self.rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])  # np matrix, not arr

        # calc rotated coordinates of the image corners
        image_w2, image_h2 = self.img_center
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * self.rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * self.rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * self.rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * self.rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        y_coords = [pt[1] for pt in rotated_coords]

        right_bound = max(x_coords)
        left_bound = min(x_coords)
        top_bound = min(y_coords)
        bott_bound = max(y_coords)

        new_w = abs(right_bound - left_bound)
        new_h = abs(bott_bound - top_bound)
        self.new_imsize_wh = (int(np.ceil(new_w)), int(np.ceil(new_h)))  # time to round
        self.new_img_center = wh_to_center(self.new_imsize_wh)

        # translation matrix to keep the image centred
        transl_x = self.new_img_center[0] - image_w2
        transl_y = self.new_img_center[1] - image_h2
        self.translation_mat = np.matrix([
            [1, 0, transl_x],
            [0, 1, transl_y],
            [0, 0, 1]
        ])

        self.affine_mat = (self.translation_mat * rot_mat)[0:2, :]

        # rect of source image in coords of rotated image
        src_rect = Rect.from_size(self.src_imsize_wh)
        self.source_rect = src_rect.move((transl_x, transl_y)).round()

        # must be rounded in same direction
        if self.source_rect.width != src_rect.width or self.source_rect.height != src_rect.height:
            raise RuntimeError('ImgRotator assert failed')

        self._calc_inner_crop()

    # with arr Nx2, i.e. (x, y)
    def transform_coords(self, arr):
        arr -= self.img_center
        tmp = arr[:, ::-1]  # to (y, x)
        tmp = (tmp * self.rot_mat_notranslate).A
        tmp = tmp[:, ::-1]  # to (x, y)
        arr[:] = tmp[:]  # in-place
        arr += self.new_img_center

    def rotate_img(self, img, use_inter_nearest):
        if use_inter_nearest:
            interp = cv2.INTER_NEAREST  # @TODO: cv2 INTER_NEAREST may shift coords, what to do?
        else:
            interp = cv2.INTER_LANCZOS4
        res = cv2.warpAffine(src=img, M=self.affine_mat, dsize=self.new_imsize_wh, flags=interp)
        return res
