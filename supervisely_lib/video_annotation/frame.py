# coding: utf-8

from supervisely_lib._utils import take_with_default
from supervisely_lib.video_annotation.constants import FIGURES, INDEX
from supervisely_lib.video_annotation.video_figure import VideoFigure
from supervisely_lib.collection.key_indexed_collection import KeyObject


class Frame(KeyObject):
    def __init__(self, index, figures=None):
        self._index = index
        self._figures = take_with_default(figures, [])

    @property
    def index(self):
        return self._index

    def key(self):
        return self._index

    @property
    def figures(self):
        return self._figures.copy()

    def validate_figures_bounds(self, img_size=None):
        if img_size is None:
            return
        for figure in self._figures:
            figure.validate_bounds(img_size, _auto_correct=False)

    def to_json(self, key_id_map=None):
        data_json = {
            INDEX: self.index,
            FIGURES: [figure.to_json(key_id_map) for figure in self.figures]
        }
        return data_json

    @classmethod
    def from_json(cls, data, objects, frames_count=None, key_id_map=None):
        index = data[INDEX]
        if index < 0:
            raise ValueError("Frame Index have to be >= 0")

        if frames_count is not None:
            if index > frames_count:
                raise ValueError("Video contains {} frames. Frame index is {}".format(frames_count, index))

        figures = []
        for figure_json in data.get(FIGURES, []):
            figure = VideoFigure.from_json(figure_json, objects, index, key_id_map)
            figures.append(figure)
        return cls(index=index, figures=figures)

    def clone(self, index=None, figures=None):
        return self.__class__(index=take_with_default(index, self.index),
                              figures=take_with_default(figures, self.figures))
