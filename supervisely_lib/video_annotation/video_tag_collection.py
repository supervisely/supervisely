# coding: utf-8

from supervisely_lib.annotation.tag_collection import TagCollection
from supervisely_lib.video_annotation.video_tag import VideoTag


class VideoTagCollection(TagCollection):
    item_type = VideoTag

    def to_json(self, key_id_map=None):
        return [tag.to_json(key_id_map) for tag in self]

    @classmethod
    def from_json(cls, data, tag_meta_collection, key_id_map=None):
        tags = [cls.item_type.from_json(tag_json, tag_meta_collection, key_id_map) for tag_json in data]
        return cls(tags)