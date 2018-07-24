# coding: utf-8

import os.path as osp

import cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib import logger


class TableImg:
    # sizes as WxH
    def __init__(self, item_size, table_size, bound_w=1):
        self.item_size = np.array(item_size)
        self.table_size = np.array(table_size)
        self.bound_w = bound_w

        img_size = self.table_size * (self.item_size + bound_w) + bound_w
        self.img = np.zeros((img_size[1], img_size[0], 3), np.uint8)
        self.img[:, :] = (0, 255, 0)

    def paste_item(self, new_img, row, col):
        img_xy = np.array((col, row)) * (self.item_size + self.bound_w) + self.bound_w

        curr_size = np.array((min(self.item_size[0], new_img.shape[1]),
                              min(self.item_size[1], new_img.shape[0])
                              ))
        img_xy_max = img_xy + curr_size
        self.img[img_xy[1]:img_xy_max[1], img_xy[0]:img_xy_max[0], :] = \
            new_img[:curr_size[1], :curr_size[0], :]


# applicable for pytorch batches
class DebugSaver:
    def __init__(self, odir, prob, target_multi, out_scale=1):
        self.odir = odir
        self.prob = prob
        self.target_multi = target_multi
        self.out_scale = out_scale
        sly.mkdir(self.odir)
        self._next_idx = 0

    @classmethod
    def to_np_img(cls, pytorch_img):
        res = np.transpose(pytorch_img.numpy(), (1, 2, 0))
        return res

    def process(self, x, target, output):
        batch_sz = x.size(0)
        for batch_idx in range(batch_sz):
            if np.random.rand() < self.prob:
                self._process_sample(
                    self.to_np_img(x[batch_idx]),
                    self.to_np_img(target[batch_idx]),
                    self.to_np_img(output[batch_idx])
                )

    def _process_sample(self, x, target, output):
        timg = TableImg((x.shape[1], x.shape[0]), (2, 2))

        bounds = np.min(x), np.max(x)
        x = (x - bounds[0]) / (bounds[1] - bounds[0]) * 255
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        timg.paste_item(x, 0, 0)
        timg.paste_item(x, 0, 1)

        sq_target = np.clip(target * self.target_multi, 0, 255)
        timg.paste_item(sq_target, 1, 0)

        sq_output = np.expand_dims(np.argmax(output, axis=2), axis=2)
        sq_output = np.clip(sq_output * self.target_multi, 0, 255)
        timg.paste_item(sq_output, 1, 1)

        out_img = timg.img
        if self.out_scale != 1:
            out_img = sly.resize_inter_nearest(out_img, fx=self.out_scale, fy=self.out_scale)

        ofpath = osp.join(self.odir, '{:08}.png'.format(self._next_idx))
        cv2.imwrite(ofpath, out_img)
        self._next_idx += 1
        logger.trace('Saved debug patch: {}'.format(ofpath))
