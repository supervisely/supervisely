# coding: utf-8
from supervisely_lib.video_annotation.video_object import VideoObject
from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection
from supervisely_lib.project.project_meta import ProjectMeta


class VideoObjectCollection(KeyIndexedCollection):
    item_type = VideoObject

    def to_json(self, key_id_map=None):
        return [item.to_json(key_id_map) for item in self]

    @classmethod
    def from_json(cls, data, project_meta: ProjectMeta, key_id_map=None):
        objects = [cls.item_type.from_json(video_object_json, project_meta, key_id_map) for video_object_json in data]
        return cls(objects)

    def __str__(self):
        return 'Objects:\n' + super(VideoObjectCollection, self).__str__()