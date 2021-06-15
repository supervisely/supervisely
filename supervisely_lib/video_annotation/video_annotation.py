# coding: utf-8

from copy import deepcopy
import uuid

from supervisely_lib._utils import take_with_default
from supervisely_lib.video_annotation.video_tag_collection import VideoTagCollection
from supervisely_lib.video_annotation.video_object_collection import VideoObjectCollection
from supervisely_lib.video_annotation.frame_collection import FrameCollection
from supervisely_lib.video_annotation.constants import FRAMES, IMG_SIZE, IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, \
                                                       DESCRIPTION, FRAMES_COUNT, TAGS, OBJECTS, VIDEO_ID, KEY, \
                                                       VIDEOS_MAP, VIDEO_NAME
from supervisely_lib.video_annotation.key_id_map import KeyIdMap


class VideoAnnotation:
    '''
    This is a class for creating and using annotations for videos
    '''
    def __init__(self, img_size, frames_count, objects=None, frames=None, tags=None, description="", key=None):
        '''
        The constructor for VideoAnnotation class.
        :param img_size: size of the image(tuple or list of integers)
        :param frames_count: int
        :param objects: VideoObjectCollection
        :param frames: FrameCollection
        :param tags: VideoTagCollection
        :param description: str
        :param key: uuid class object
        '''
        if not isinstance(img_size, (tuple, list)):
            raise TypeError('{!r} has to be a tuple or a list. Given type "{}".'.format('img_size', type(img_size)))
        self._img_size = tuple(img_size)
        self._frames_count = frames_count

        self._description = description
        self._tags = take_with_default(tags, VideoTagCollection())
        self._objects = take_with_default(objects, VideoObjectCollection())
        self._frames = take_with_default(frames, FrameCollection())
        self._key = take_with_default(key, uuid.uuid4())

        self.validate_figures_bounds()

    @property
    def img_size(self):
        return deepcopy(self._img_size)

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
        '''
        :return: list of figures from all frames in collection
        '''
        return self.frames.figures

    @property
    def tags(self):
        return self._tags

    def key(self):
        return self._key

    @property
    def description(self):
        return self._description

    def validate_figures_bounds(self):
        '''
        The function validate_figures_bounds checks if image contains figures from all frames in collection. Raise error if figure is out of image bounds
        '''
        for frame in self.frames:
            frame.validate_figures_bounds(self.img_size)

    def to_json(self, key_id_map: KeyIdMap=None):
        '''
        The function to_json convert videoannotation to json format
        :param key_id_map: KeyIdMap class object
        :return: videoannotation in json format
        '''
        res_json = {
                        IMG_SIZE: {
                                    IMG_SIZE_HEIGHT: int(self.img_size[0]),
                                    IMG_SIZE_WIDTH: int(self.img_size[1])
                                  },
                        DESCRIPTION: self.description,
                        KEY: self.key().hex,
                        TAGS: self.tags.to_json(key_id_map),
                        OBJECTS: self.objects.to_json(key_id_map),
                        FRAMES: self.frames.to_json(key_id_map),
                        FRAMES_COUNT: self.frames_count
                    }

        if key_id_map is not None:
            video_id = key_id_map.get_video_id(self.key())
            if video_id is not None:
                res_json[VIDEO_ID] = video_id

        return res_json

    @classmethod
    def from_json(cls, data, project_meta, key_id_map: KeyIdMap=None):
        '''
        The function from_json convert videoannotation from json format to VideoAnnotation class object.
        :param data: input videoannotation in json format
        :param project_meta: ProjectMeta class object
        :param key_id_map: KeyIdMap class object
        :return: VideoAnnotation class object
        '''
        #video_name = data[VIDEO_NAME]
        video_key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_video(video_key, data.get(VIDEO_ID, None))

        img_size_dict = data[IMG_SIZE]
        img_height = img_size_dict[IMG_SIZE_HEIGHT]
        img_width = img_size_dict[IMG_SIZE_WIDTH]
        img_size = (img_height, img_width)

        description = data.get(DESCRIPTION, "")
        frames_count = data[FRAMES_COUNT]

        tags = VideoTagCollection.from_json(data[TAGS], project_meta.tag_metas, key_id_map)
        objects = VideoObjectCollection.from_json(data[OBJECTS], project_meta, key_id_map)
        frames = FrameCollection.from_json(data[FRAMES], objects, frames_count, key_id_map)

        return cls(img_size=img_size,
                   frames_count=frames_count,
                   objects=objects,
                   frames=frames,
                   tags=tags,
                   description=description,
                   key=video_key)

    def clone(self, img_size=None, frames_count=None, objects=None, frames=None, tags=None, description=None):
        '''
        :param img_size: size of the image(tuple or list of integers)
        :param frames_count: int
        :param objects: VideoObjectCollection
        :param frames: FrameCollection
        :param tags: VideoTagCollection
        :param description: str
        :return: VideoAnnotation class object
        '''
        return VideoAnnotation(img_size=take_with_default(img_size, self.img_size),
                               frames_count=take_with_default(frames_count, self.frames_count),
                               objects=take_with_default(objects, self.objects),
                               frames=take_with_default(frames, self.frames),
                               tags=take_with_default(tags, self.tags),
                               description=take_with_default(description, self.description))


    def is_empty(self):
        if len(self.objects) == 0 and len(self.tags) == 0:
            return True
        else:
            return False
