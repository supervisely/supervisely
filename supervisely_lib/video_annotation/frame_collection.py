# coding: utf-8

from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection
from supervisely_lib.video_annotation.frame import Frame
from supervisely_lib.api.module_api import ApiField


class FrameCollection(KeyIndexedCollection):
    '''
    Collection that stores Frame instances.
    '''
    item_type = Frame

    def to_json(self, key_id_map=None):
        '''
        Converts collection to json serializable list.
        :param key_id_map: KeyIdMap class object
        :return: list of frames in json format
        '''
        return [frame.to_json(key_id_map) for frame in self]

    @classmethod
    def from_json(cls, data, objects, frames_count=None, key_id_map=None):
        '''
        Creates collection from json serializable list
        :param data: list of frames in json format
        :param objects: VideoObjectCollection
        :param frames_count: int
        :param key_id_map: int
        :return: FrameCollection class object
        '''
        frames = [cls.item_type.from_json(frame_json, objects, frames_count, key_id_map) for frame_json in data]
        return cls(frames)

    def __str__(self):
        return 'Frames:\n' + super(FrameCollection, self).__str__()

    @property
    def figures(self):
        '''
        :return: list of figures from all frames in collection
        '''
        figures_array = []
        for frame in self:
            figures_array.extend(frame.figures)
        return figures_array

    def get_figures_and_keys(self, key_id_map):
        '''
        :param key_id_map: KeyIdMap class object
        :return: list of figures from all frames in collection in json format, list of keys from all figures in frames in collection
        '''
        keys = []
        figures_json = []
        for frame in self:
            for figure in frame.figures:
                keys.append(figure.key())
                figure_json = figure.to_json(key_id_map)
                figure_json[ApiField.META] = {ApiField.FRAME: frame.index}
                figures_json.append(figure_json)
        return figures_json, keys

