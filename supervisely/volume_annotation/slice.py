# coding: utf-8

from supervisely.video_annotation.frame import Frame
from supervisely.volume_annotation.volume_figure import VolumeFigure


class Slice(Frame):
    figure_type = VolumeFigure

    @classmethod
    def from_json(cls, data, objects, slices_count=None, key_id_map=None):
        _frame = super().from_json(data, objects, slices_count, key_id_map)
        return cls(index=_frame.index, figures=_frame.figures)
