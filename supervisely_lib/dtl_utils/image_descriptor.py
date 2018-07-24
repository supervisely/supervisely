# coding: utf-8

import cv2
import numpy as np

from ..utils import ensure_base_path


class ImageDescriptor:
    def __init__(self, info):
        self.image_data = None  # can be changed in comp graph
        self.info = info
        self.res_ds_name = '{}__{}'.format(self.info.project_name, self.info.ds_name)

    def read_image(self):
        if self.image_data is not None:
            return self.image_data

        img = cv2.imread(self.info.img_path)
        if img is None:
            raise RuntimeError('Image not found. {}'.format(self.info.img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def update_image(self, img):
        self.image_data = img

    def write_image_local(self, img_path):
        if self.image_data is None:
            raise RuntimeError("ImageDescriptor [write_image_local] image_data is None:{}".format(img_path))
        img_res = self.image_data.astype(np.uint8)
        img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
        ensure_base_path(img_path)
        cv2.imwrite(img_path, img_res)

    def encode_image(self):
        if self.image_data is None:
            raise RuntimeError("ImageDescriptor [encode_image] image_data is None.")
        img_res = self.image_data.astype(np.uint8)
        img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
        res_bytes = cv2.imencode('.png', img_res)[1]
        return res_bytes

    def need_write(self):
        if self.image_data is None:
            return False
        return True

    def clone_with_img(self, new_img):
        new_obj = self.__class__(self.info)
        new_obj.image_data = new_img
        new_obj.res_ds_name = self.res_ds_name
        return new_obj

    def get_res_ds_name(self):
        return self.res_ds_name

    def get_image_ext(self):
        return self.info.ia_data['image_ext']

    def get_pr_name(self):
        return self.info.project_name

    def get_ds_name(self):
        return self.info.ds_name

    def get_img_name(self):
        return self.info.image_name

    def get_img_path(self):
        return self.info.img_path
