# coding: utf-8

import uuid

from supervisely_lib._utils import take_with_default
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
from supervisely_lib.video_annotation.constants import FRAMES, DESCRIPTION, FRAMES_COUNT, TAGS, OBJECTS, KEY
from supervisely_lib.video_annotation.frame_collection import FrameCollection
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.video_annotation.video_annotation import VideoAnnotation
from supervisely_lib.pointcloud_annotation.pointcloud_annotation import PointcloudAnnotation
from supervisely_lib.video_annotation.video_tag_collection import VideoTagCollection


class PointcloudEpisodeAnnotation(VideoAnnotation):
    """
        This is a class for creating and using PointcloudEpisodeAnnotation
    """
    DATASET_ID_KEY = 'datasetId'

    def __init__(self, frames_count=None, objects=None, frames=None, tags=None, description="", key=None):
        """
        :param frames_count: int
        :param objects: PointcloudObjectCollection
        :param frames: FrameCollection
        :param tags: VideoTagCollection
        :param description: str
        :param key: uuid class object
        """

        self._frames_count = frames_count
        self._description = description
        self._frames = take_with_default(frames, FrameCollection())
        self._tags = take_with_default(tags, VideoTagCollection())
        self._objects = take_with_default(objects, PointcloudObjectCollection())
        self._key = take_with_default(key, uuid.uuid4())

    def to_json(self, key_id_map: KeyIdMap=None):
        '''
        The function to_json convert PointcloudEpisodeAnnotation to json format
        :param key_id_map: KeyIdMap class object
        :return: PointcloudEpisodeAnnotation in json format
        '''
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
                res_json[self.DATASET_ID_KEY] = dataset_id

        return res_json

    @classmethod
    def from_json(cls, data, project_meta, key_id_map: KeyIdMap=None):
        """
        :param data: input PointcloudEpisodeAnnotation in json format
        :param project_meta: ProjectMeta class object
        :param key_id_map: KeyIdMap class object
        :return: PointcloudEpisodeAnnotation class object
        """

        item_key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_video(item_key, data.get(cls.DATASET_ID_KEY, None))

        description = data.get(DESCRIPTION, "")
        frames_count = data.get(FRAMES_COUNT, 0)

        tags = VideoTagCollection.from_json(data[TAGS], project_meta.tag_metas, key_id_map)
        objects = PointcloudObjectCollection.from_json(data[OBJECTS], project_meta, key_id_map)
        frames = FrameCollection.from_json(data[FRAMES], objects, key_id_map=key_id_map)

        return cls(frames_count, objects, frames, tags, description, item_key)

    def clone(self, frames_count=None, objects=None, frames=None, tags=None, description=""):
        """
        :param frames_count: int
        :param frames: FrameCollection
        :param objects: PointcloudObjectCollection
        :param tags: VideoTagCollection
        :param description: str
        :return: PointcloudEpisodeAnnotation class object
        """
        return PointcloudEpisodeAnnotation(frames_count=take_with_default(frames_count, self.frames_count),
                                           objects=take_with_default(objects, self.objects),
                                           frames=take_with_default(frames, self.frames),
                                           tags=take_with_default(tags, self.tags),
                                           description=take_with_default(description, self.description))

    def get_single_annotation(self, frame_idx):
        frame = self.frames.get(frame_idx)
        figures = frame.figures
        tags = None  # TODO: make tags!
        objects = PointcloudObjectCollection(set([fig.parent_object for fig in figures]))
        return PointcloudAnnotation(objects, figures, tags)
