# coding: utf-8

from supervisely_lib._utils import take_with_default
from supervisely_lib.volume_annotation import constants as const
from supervisely_lib.volume_annotation.volume_figure import VolumeFigure
from supervisely_lib.collection.key_indexed_collection import KeyObject


class Slice(KeyObject):
    def __init__(self, index, figures=None, normal=None):
        self._index = index
        self._normal = normal
        self._figures = take_with_default(figures, [])

    @property
    def normal(self):
        return self._normal

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
    def from_json(cls, data, objects, normal, key_id_map=None):

        index = data[const.INDEX]
        if index < 0:
            raise ValueError("Frame Index have to be >= 0")

        figures = []
        fig_meta = {const.SLICE_INDEX: index, const.NORMAL: normal}
        for figure_json in data.get(const.FIGURES, []):
            figure = VolumeFigure.from_json(figure_json, objects, fig_meta, key_id_map)
            figures.append(figure)
        return cls(index=index, figures=figures, normal=normal)

    def clone(self, index=None, figures=None, normal=None):
        return self.__class__(index=take_with_default(index, self.index),
                              figures=take_with_default(figures, self.figures),
                              normal=take_with_default(normal, self.normal))
