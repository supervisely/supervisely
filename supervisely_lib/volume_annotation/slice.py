# coding: utf-8

from supervisely_lib._utils import take_with_default
from supervisely_lib.volume_annotation import constants as const
from supervisely_lib.volume_annotation.volume_figure import VolumeFigure
from supervisely_lib.collection.key_indexed_collection import KeyObject


class Slice(KeyObject):
    def __init__(self, index, figures=None):
        self._index = index
        self._figures = take_with_default(figures, [])

    @property
    def index(self):
        return self._index

    @property
    def figures(self):
        return self._figures.copy()

    def key(self):
        return self._index

    def to_json(self, key_id_map=None):
        '''
        The function to_json convert frame to json format
        :param key_id_map: KeyIdMap class object
        :return: frame in json format
        '''
        data_json = {
            const.INDEX: self.index,
            const.FIGURES: [figure.to_json(key_id_map) for figure in self.figures]
        }
        return data_json

    @classmethod
    def from_json(cls, data, objects, key_id_map=None):

        index = data[const.INDEX]
        if index < 0:
            raise ValueError("Frame Index have to be >= 0")

        figures = []
        for figure_json in data.get(const.FIGURES, []):
            figure = VolumeFigure.from_json(figure_json, objects, index, key_id_map)
            figures.append(figure)
        return cls(index=index, figures=figures)

    def clone(self, index=None, figures=None):
        return self.__class__(index=take_with_default(index, self.index),
                              figures=take_with_default(figures, self.figures))
