# coding: utf-8

from supervisely_lib.annotation.tag_collection import TagCollection
from supervisely_lib.video_annotation.video_tag import VideoTag


class VideoTagCollection(TagCollection):
    '''
    Collection that stores VideoTag instances.
    '''
    item_type = VideoTag

    def to_json(self, key_id_map=None):
        '''
        Converts collection to json serializable list.
        :param key_id_map: KeyIdMap class object
        :return: list of VideoTag in json format
        '''
        return [tag.to_json(key_id_map) for tag in self]

    @classmethod
    def from_json(cls, data, tag_meta_collection, key_id_map=None):
        '''
        Creates collection from json serializable list
        :param data: list of VideoTags in json format
        :param tag_meta_collection: TagMetaCollection
        :param key_id_map: KeyIdMap class object
        :return: VideoTagCollection class object
        '''
        tags = [cls.item_type.from_json(tag_json, tag_meta_collection, key_id_map) for tag_json in data]
        return cls(tags)