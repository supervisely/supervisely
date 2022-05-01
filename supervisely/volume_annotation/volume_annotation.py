# coding: utf-8

from copy import deepcopy
import uuid

from supervisely._utils import take_with_default
from supervisely.volume_annotation.volume_tag_collection import VolumeTagCollection
from supervisely.volume_annotation.volume_object_collection import (
    VolumeObjectCollection,
)
from supervisely.volume_annotation.plane import Plane
from supervisely.volume_annotation.plane_info import (
    PlaneName,
    get_img_size_from_volume_meta,
    get_slices_count_from_volume_meta,
)
from supervisely.volume_annotation.constants import (
    FRAMES,
    IMG_SIZE,
    IMG_SIZE_HEIGHT,
    IMG_SIZE_WIDTH,
    DESCRIPTION,
    FRAMES_COUNT,
    TAGS,
    OBJECTS,
    VIDEO_ID,
    KEY,
    VIDEOS_MAP,
    VIDEO_NAME,
)
from supervisely.video_annotation.key_id_map import KeyIdMap


class VolumeAnnotation:
    def __init__(
        self,
        volume_meta,
        objects=None,
        plane_sagittal=None,
        plane_coronal=None,
        plane_axial=None,
        tags=None,
        key=None,
    ):
        self._volume_meta = volume_meta
        self._tags = take_with_default(tags, VolumeTagCollection())
        self._objects = take_with_default(objects, VolumeObjectCollection())
        self._key = take_with_default(key, uuid.uuid4())

        self._plane_sagittal = take_with_default(
            plane_sagittal,
            Plane(PlaneName.SAGITTAL, volume_meta=volume_meta),
        )

        self._plane_coronal = take_with_default(
            plane_coronal,
            Plane(PlaneName.CORONAL, volume_meta=volume_meta),
        )

        self._plane_axial = take_with_default(
            plane_axial,
            Plane(PlaneName.AXIAL, volume_meta=volume_meta),
        )

        self.validate_figures_bounds()

    @property
    def volume_meta(self):
        return deepcopy(self._volume_meta)

    @property
    def plane_sagittal(self):
        return self._plane_sagittal

    @property
    def plane_coronal(self):
        return self._plane_coronal

    @property
    def plane_axial(self):
        return self._plane_axial

    @property
    def objects(self):
        return self._objects

    @property
    def tags(self):
        return self._tags

    def key(self):
        return self._key

    def validate_figures_bounds(self):
        raise NotImplementedError()
        # for frame in self.frames:
        #     frame.validate_figures_bounds(self.img_size)

    def is_empty(self):
        if len(self.objects) == 0 and len(self.tags) == 0:
            return True
        else:
            return False

    def clone(
        self,
        volume_meta,
        objects=None,
        plane_sagittal=None,
        plane_coronal=None,
        plane_axial=None,
        tags=None,
        key=None,
    ):
        return VideoAnnotation(
            # volume_meta=take_with_default(take_with_default, self.volume_meta)
            # img_size=take_with_default(img_size, self.img_size),
            # frames_count=take_with_default(frames_count, self.frames_count),
            # objects=take_with_default(objects, self.objects),
            # frames=take_with_default(frames, self.frames),
            # tags=take_with_default(tags, self.tags),
            # description=take_with_default(description, self.description),
        )


class VideoAnnotation:
    def to_json(self, key_id_map: KeyIdMap = None):
        """
        The function to_json convert videoannotation to json format
        :param key_id_map: KeyIdMap class object
        :return: videoannotation in json format
        """
        res_json = {
            IMG_SIZE: {
                IMG_SIZE_HEIGHT: int(self.img_size[0]),
                IMG_SIZE_WIDTH: int(self.img_size[1]),
            },
            DESCRIPTION: self.description,
            KEY: self.key().hex,
            TAGS: self.tags.to_json(key_id_map),
            OBJECTS: self.objects.to_json(key_id_map),
            FRAMES: self.frames.to_json(key_id_map),
            FRAMES_COUNT: self.frames_count,
        }

        if key_id_map is not None:
            video_id = key_id_map.get_video_id(self.key())
            if video_id is not None:
                res_json[VIDEO_ID] = video_id

        return res_json

    @classmethod
    def from_json(cls, data, project_meta, key_id_map: KeyIdMap = None):
        """
        The function from_json convert videoannotation from json format to VideoAnnotation class object.
        :param data: input videoannotation in json format
        :param project_meta: ProjectMeta class object
        :param key_id_map: KeyIdMap class object
        :return: VideoAnnotation class object
        """
        # video_name = data[VIDEO_NAME]
        video_key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_video(video_key, data.get(VIDEO_ID, None))

        img_size_dict = data[IMG_SIZE]
        img_height = img_size_dict[IMG_SIZE_HEIGHT]
        img_width = img_size_dict[IMG_SIZE_WIDTH]
        img_size = (img_height, img_width)

        description = data.get(DESCRIPTION, "")
        frames_count = data[FRAMES_COUNT]

        tags = VideoTagCollection.from_json(
            data[TAGS], project_meta.tag_metas, key_id_map
        )
        objects = VideoObjectCollection.from_json(
            data[OBJECTS], project_meta, key_id_map
        )
        frames = FrameCollection.from_json(
            data[FRAMES], objects, frames_count, key_id_map
        )

        return cls(
            img_size=img_size,
            frames_count=frames_count,
            objects=objects,
            frames=frames,
            tags=tags,
            description=description,
            key=video_key,
        )
