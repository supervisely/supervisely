# coding: utf-8

from .rectangle import Rect


# to get rects of sliding window from source parameters
class SlidingWindows:
    # params in integer units (i.e. pixels)
    def __init__(self, window_wh, min_overlap_xy):
        self.window_wh = tuple(window_wh)
        self.min_overlap_xy = tuple(min_overlap_xy)
        self.stride_xy = tuple(self.window_wh[i] - self.min_overlap_xy[i] for i in (0, 1))
        if min(self.stride_xy) < 1:
            raise RuntimeError('Wrong sliding window settings, overlap is too high.')

    def get(self, source_wh):
        source_rect = Rect.from_size(source_wh)
        if not source_rect.contains(Rect.from_size(self.window_wh)):
            raise RuntimeError('Sliding window: window is larger than source (image).')

        wh_limit = (source_wh[0] - self.window_wh[0], source_wh[1] - self.window_wh[1])
        for wind_top in range(0, wh_limit[1] + self.stride_xy[1], self.stride_xy[1]):
            for wind_left in range(0, wh_limit[0] + self.stride_xy[0], self.stride_xy[0]):
                window_lt = (
                    min(wind_left, wh_limit[0]),
                    min(wind_top, wh_limit[1]),
                )  # shift to preserve window size
                roi = Rect.from_size(self.window_wh)
                roi = roi.move(window_lt)
                if roi.is_empty or (not source_rect.contains(roi)):
                    raise RuntimeError('Sliding window: result crop bounds are invalid.')
                yield roi
