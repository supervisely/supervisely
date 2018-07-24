# coding: utf-8

from copy import deepcopy

import cv2
import numpy as np

from .abstract_figure import AbstractFigure


class AbstractVectorFigure(AbstractFigure):
    COMMON_PTS_DTYPE = np.float64

    def _get_points_stacked(self):
        exterior, interiors = self._get_points()
        res = np.vstack([exterior] + interiors)
        return res

    def _get_points(self):
        pts_dict = self.data['points']
        return pts_dict['exterior'], pts_dict['interior']

    def _set_points(self, exterior, interiors=None):
        self.data['points'] = {
            'exterior': exterior,
            'interior': interiors if interiors is not None else []
        }

    def _approx_ring_dp(self, ring, epsilon, closed):
        new_ring = cv2.approxPolyDP(ring.astype(np.int32), epsilon, closed)
        new_ring = np.squeeze(new_ring, 1)
        new_ring = self.ring_to_np_points(new_ring)
        return new_ring

    @classmethod
    def ring_to_np_points(cls, r):
        return np.array(r, dtype=cls.COMMON_PTS_DTYPE)

    # fn must transform np array of pts with shape Nx2 in-place (sic!)
    def transform_points(self, fn):
        exterior, interiors = self._get_points()
        fn(exterior)
        for interior in interiors:
            fn(interior)

    def rotate(self, rotator):
        fn = rotator.transform_coords
        self.transform_points(fn)

    def resize(self, resizer):
        fn = resizer.transform_coords
        self.transform_points(fn)

    def shift(self, delta):
        def move_fn(x):
            x[:] = x + delta
        self.transform_points(move_fn)

    def flip(self, is_horiz, img_wh):
        w, h = img_wh
        if is_horiz:
            def flip_fn(x):
                x[:, 1] = h - x[:, 1]  # h - Ys
        else:
            def flip_fn(x):
                x[:, 0] = w - x[:, 0]  # w - Xs
        self.transform_points(flip_fn)

    # Douglas-Pecker for vector figures only
    def approx_dp(self, epsilon):
        pass  # ok for simplest figures

    # default implementation for figures w/out interiors
    def pack(self):
        # convert np arrays to lists of lists
        exterior, _ = self._get_points()
        packed = deepcopy(self.data)
        packed['points'] = {
            'exterior': exterior.tolist(),
            'interior': []
        }
        return packed
