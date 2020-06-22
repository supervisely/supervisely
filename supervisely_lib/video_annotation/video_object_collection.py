# coding: utf-8
from supervisely_lib.video_annotation.video_object import VideoObject
from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection
from supervisely_lib.project.project_meta import ProjectMeta


class VideoObjectCollection(KeyIndexedCollection):
    '''
    Collection that stores VideoObject instances.
    '''
    item_type = VideoObject

    def to_json(self, key_id_map=None):
        '''
        Converts collection to json serializable list.
        :param key_id_map: KeyIdMap class object
        :return: list of VideoObjects in json format
        '''
        return [item.to_json(key_id_map) for item in self]

    @classmethod
    def from_json(cls, data, project_meta: ProjectMeta, key_id_map=None):
        '''
        Creates collection from json serializable list
        :param data: list of VideoObjects in json format
        :param project_meta: ProjectMeta class object
        :param key_id_map: KeyIdMap class object
        :return: VideoObjectCollection class object
        '''
        objects = [cls.item_type.from_json(video_object_json, project_meta, key_id_map) for video_object_json in data]
        return cls(objects)

    def __str__(self):
        return 'Objects:\n' + super(VideoObjectCollection, self).__str__()