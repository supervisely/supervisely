# coding: utf-8

import uuid

from supervisely._utils import take_with_default
from supervisely.api.module_api import ApiField
from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
from supervisely.video_annotation.constants import FRAMES, DESCRIPTION, FRAMES_COUNT, TAGS, OBJECTS, KEY
from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_tag_collection import VideoTagCollection


class PointcloudEpisodeAnnotation:
    def __init__(self, frames_count=None, objects=None, frames=None, tags=None, description="", key=None):
        self._frames_count = frames_count
        self._description = description
        self._frames = take_with_default(frames, FrameCollection())
        self._tags = take_with_default(tags, VideoTagCollection())
        self._objects = take_with_default(objects, PointcloudObjectCollection())
        self._key = take_with_default(key, uuid.uuid4())

    def to_json(self, key_id_map: KeyIdMap = None):
        res_json = {
            DESCRIPTION: self.description,
            KEY: self.key().hex,
            TAGS: self.tags.to_json(key_id_map),
            OBJECTS: self.objects.to_json(key_id_map),
            FRAMES_COUNT: self.frames_count,
            FRAMES: self.frames.to_json(key_id_map),
        }

        if key_id_map is not None:
            dataset_id = key_id_map.get_video_id(self.key())
            if dataset_id is not None:
                res_json[ApiField.DATASET_ID] = dataset_id

        return res_json

    @classmethod
    def from_json(cls, data, project_meta, key_id_map: KeyIdMap = None):
        item_key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_video(item_key, data.get(ApiField.DATASET_ID, None))

        description = data.get(DESCRIPTION, "")
        frames_count = data.get(FRAMES_COUNT, 0)

        tags = VideoTagCollection.from_json(data[TAGS], project_meta.tag_metas, key_id_map)
        objects = PointcloudObjectCollection.from_json(data[OBJECTS], project_meta, key_id_map)
        frames = FrameCollection.from_json(data[FRAMES], objects, key_id_map=key_id_map)

        return cls(frames_count, objects, frames, tags, description, item_key)

    def clone(self, frames_count=None, objects=None, frames=None, tags=None, description=""):
        return PointcloudEpisodeAnnotation(frames_count=take_with_default(frames_count, self.frames_count),
                                           objects=take_with_default(objects, self.objects),
                                           frames=take_with_default(frames, self.frames),
                                           tags=take_with_default(tags, self.tags),
                                           description=take_with_default(description, self.description))

    @property
    def frames_count(self):
        return self._frames_count

    @property
    def objects(self):
        return self._objects

    @property
    def frames(self):
        return self._frames

    @property
    def figures(self):
        return self.frames.figures

    @property
    def tags(self):
        return self._tags

    def key(self):
        return self._key

    @property
    def description(self):
        return self._description

    def is_empty(self):
        if len(self.objects) == 0 and len(self.tags) == 0:
            return True
        else:
            return False
