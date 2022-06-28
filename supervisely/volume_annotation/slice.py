# coding: utf-8

from supervisely.video_annotation.frame import Frame
from supervisely.volume_annotation.volume_figure import VolumeFigure
from supervisely.volume_annotation.constants import FIGURES, INDEX


class Slice(Frame):
    figure_type = VolumeFigure

    @classmethod
    def from_json(cls, data, objects, slices_count=None, key_id_map=None):
        raise NotImplementedError()
        _frame = super().from_json(data, objects, slices_count, key_id_map)
        return cls(index=_frame.index, figures=_frame.figures)

    @classmethod
    def from_json(cls, data, objects, plane_name, slices_count=None, key_id_map=None):
        index = data[INDEX]
        if index < 0:
            raise ValueError("Frame Index have to be >= 0")

        if slices_count is not None:
            if index > slices_count:
                raise ValueError(
                    "Item contains {} frames. Frame index is {}".format(
                        slices_count, index
                    )
                )

        figures = []
        for figure_json in data.get(FIGURES, []):
            figure = cls.figure_type.from_json(
                figure_json, objects, plane_name, index, key_id_map
            )
            figures.append(figure)
        return cls(index=index, figures=figures)
