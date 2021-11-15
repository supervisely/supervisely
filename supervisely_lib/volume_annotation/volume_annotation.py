import uuid

from supervisely_lib._utils import take_with_default
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.video_annotation.video_tag_collection import VideoTagCollection
from supervisely_lib.volume_annotation.volume_object_collection import VolumeObjectCollection
from supervisely_lib.volume_annotation.plane_collection import PlaneCollection

from supervisely_lib.volume_annotation.constants import VOLUME_META, OBJECTS, KEY, VOLUME_ID, DESCRIPTION, TAGS, PLANES


class VolumeAnnotation:
    def __init__(self,
                 volume_meta: dict,
                 objects: VolumeObjectCollection,
                 planes: PlaneCollection,
                 tags: VideoTagCollection = None,
                 description: str = "",
                 key=None):

        self._volume_meta = volume_meta
        self._objects = take_with_default(objects, VolumeObjectCollection())
        self._planes = take_with_default(planes, PlaneCollection())
        self._tags = take_with_default(tags, VideoTagCollection())
        self._description = description
        self._key = take_with_default(key, uuid.uuid4())

    @property
    def volume_meta(self):
        return self._volume_meta

    @property
    def planes(self):
        return self._planes

    @property
    def objects(self):
        return self._objects

    @property
    def tags(self):
        return self._tags

    @property
    def description(self):
        return self._description

    @property
    def figures(self):
        return self.planes.figures

    def key(self):
        return self._key

    @classmethod
    def from_json(cls, data, project_meta, key_id_map: KeyIdMap = None):
        volume_key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_video(volume_key, data.get(VOLUME_ID, None))

        description = data.get(DESCRIPTION, "")
        volume_meta = data.get(VOLUME_META, None)
        tags = VideoTagCollection.from_json(data[TAGS], project_meta.tag_metas, key_id_map)
        objects = VolumeObjectCollection.from_json(data[OBJECTS], project_meta, key_id_map)
        planes = PlaneCollection.from_json(data[PLANES], objects, key_id_map)

        return cls(volume_meta=volume_meta,
                   objects=objects,
                   planes=planes,
                   tags=tags,
                   description=description,
                   key=volume_key)

    def to_json(self, key_id_map: KeyIdMap = None):
        res_json = {
            VOLUME_META: self.volume_meta,
            DESCRIPTION: self.description,
            KEY: self.key().hex,
            TAGS: self.tags.to_json(key_id_map),
            OBJECTS: self.objects.to_json(key_id_map),
            PLANES: self.planes.to_json(key_id_map),
        }

        if key_id_map is not None:
            volume_id = key_id_map.get_video_id(self.key())
            if volume_id is not None:
                res_json[VOLUME_ID] = volume_id
        return res_json

    def clone(self, volume_meta=None, objects=None, planes=None, tags=None, description=None):
        return VolumeAnnotation(volume_meta=take_with_default(volume_meta, self.volume_meta),
                                planes=take_with_default(planes, self.planes),
                                objects=take_with_default(objects, self.objects),
                                tags=take_with_default(tags, self.tags),
                                description=take_with_default(description, self.description))
