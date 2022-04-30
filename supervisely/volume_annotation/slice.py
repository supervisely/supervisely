# coding: utf-8

from supervisely.video_annotation.frame import Frame
from supervisely._utils import take_with_default
from supervisely.volume_annotation.constants import PLANE_NAME
from supervisely.volume_annotation.volume_figure import VolumeFigure
from supervisely.volume_annotation.plane_info import PlaneName, get_normal


class Slice(Frame):
    figure_type = VolumeFigure

    def __init__(self, plane_name, index, figures=None):
        PlaneName.validate(plane_name)
        self._plane_name = plane_name
        super().__init__(index, figures)

    @property
    def plane_name(self):
        return self._plane_name

    @property
    def normal(self):
        return get_normal(self.plane_name)

    @classmethod
    def from_json(cls, data, objects, slices_count=None, key_id_map=None):
        plane_name = data[PLANE_NAME]
        _frame = super().from_json(data, objects, slices_count, key_id_map)
        return cls(plane_name=plane_name, index=_frame.index, figures=_frame.figures)

    def clone(self, plane_name=None, index=None, figures=None):
        return self.__class__(
            plane_name=take_with_default(plane_name, self.plane_name),
            index=take_with_default(index, self.index),
            figures=take_with_default(figures, self.figures),
        )
