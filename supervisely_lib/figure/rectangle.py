# coding: utf-8

import numpy as np


# float or integer rect; res of some operations may be empty (no excess checks)
class Rect(object):
    def __init__(self, left, top, right, bottom):
        self.data = (left, top, right, bottom)

    @property
    def width(self):
        return self.data[2] - self.data[0]

    @property
    def height(self):
        return self.data[3] - self.data[1]

    @property
    def left(self):
        return self.data[0]

    @property
    def top(self):
        return self.data[1]

    @property
    def right(self):
        return self.data[2]

    @property
    def bottom(self):
        return self.data[3]

    @property
    def point0(self):
        return self.data[0], self.data[1]

    @property
    def point1(self):
        return self.data[2], self.data[3]

    @property
    def is_empty(self):
        res = self.height <= 0 or self.width <= 0
        return res

    @property
    def area(self):
        if self.is_empty:
            return 0
        else:
            return self.width * self.height

    # float result
    @property
    def center(self):
        return ((self.data[0] + self.data[2]) / 2.0,
                (self.data[1] + self.data[3]) / 2.0,)

    def contains(self, rhs):
        r0 = self.data
        r1 = rhs.data
        res = (r0[0] <= r1[0]) and (r0[1] <= r1[1]) and \
              (r0[2] >= r1[2]) and (r0[3] >= r1[3])
        return res

    def contains_point(self, pt):  # tuple-like pt
        r0 = self.data
        res = (r0[0] <= pt[0]) and (r0[1] <= pt[1]) and \
              (r0[2] >= pt[0]) and (r0[3] >= pt[1])
        return res

    # res may be empty
    def intersection(self, rhs):
        r0 = self.data
        r1 = rhs.data
        res = Rect(
            left=max(r0[0], r1[0]),
            top=max(r0[1], r1[1]),
            right=min(r0[2], r1[2]),
            bottom=min(r0[3], r1[3])
        )
        return res

    def combination(self, rhs):
        r0 = self.data
        r1 = rhs.data
        res = Rect(
            left=min(r0[0], r1[0]),
            top=min(r0[1], r1[1]),
            right=max(r0[2], r1[2]),
            bottom=max(r0[3], r1[3])
        )
        return res

    # err if both rects are empty
    def iou(self, rhs):
        inters_area = self.intersection(rhs).area
        union_area = self.area + rhs.area - inters_area
        iou = inters_area / union_area
        return iou

    # delta as (dx, dy)
    def move(self, delta):
        r = self.data
        res = Rect(r[0] + delta[0], r[1] + delta[1],
                   r[2] + delta[0], r[3] + delta[1])
        return res

    def round(self):
        res = (int(np.floor(x + 0.5)) for x in self.data)  # floor: same direction
        res = Rect(*res)
        return res

    def to_np_points(self):
        left, top, right, bottom = self.data
        res = np.array([
            (left, top), (left, bottom), (right, bottom), (right, top)  # ccw
        ])
        return res

    @classmethod
    def from_arr(cls, nparr):
        res = cls(0, 0, nparr.shape[1], nparr.shape[0])
        return res

    @classmethod
    def from_size(cls, size_wh):
        res = cls(0, 0, size_wh[0], size_wh[1])
        return res

    @classmethod
    def from_np_points(cls, pts):
        xs = pts[:, 0]
        ys = pts[:, 1]
        res = cls(np.min(xs), np.min(ys), np.max(xs), np.max(ys))  # @TODO: to float? or it will contain np.float
        return res
