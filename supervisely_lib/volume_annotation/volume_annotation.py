import uuid

from supervisely_lib._utils import take_with_default
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.video_annotation.video_tag_collection import VideoTagCollection
from supervisely_lib.volume_annotation.volume_object_collection import VolumeObjectCollection
from supervisely_lib.volume_annotation.plane import Plane
import supervisely_lib.volume_annotation.constants as const


class VolumeAnnotation:
    def __init__(self,
                 volume_meta: dict,
                 objects: VolumeObjectCollection,
                 axial: Plane = None,
                 sagittal: Plane = None,
                 coronal: Plane = None,
                 tags: VideoTagCollection = None,
                 description: str = "",
                 key=None):

        self._volume_meta = volume_meta
        self._objects = take_with_default(objects, VolumeObjectCollection())
        self._axial = take_with_default(axial, Plane())
        self._sagittal = take_with_default(sagittal, Plane())
        self._coronal = take_with_default(coronal, Plane())
        self._tags = take_with_default(tags, VideoTagCollection())
        self._description = description
        self._key = take_with_default(key, uuid.uuid4())

    @property
    def volume_meta(self):
        return self._volume_meta

    @property
    def axial(self):
        return self._axial

    @property
    def sagittal(self):
        return self._sagittal

    @property
    def coronal(self):
        return self._coronal

    @property
    def objects(self):
        return self._objects

    @property
    def tags(self):
        return self._tags

    @property
    def description(self):
        return self._description

    def key(self):
        return self._key

    @classmethod
    def from_json(cls, data, project_meta, key_id_map: KeyIdMap = None):
        volume_key = uuid.UUID(data[const.KEY]) if const.KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_video(volume_key, data.get(const.VOLUME_ID, None))

        description = data.get(const.DESCRIPTION, "")
        volume_meta = data[const.VOLUME_META]
        tags = VideoTagCollection.from_json(data[const.TAGS], project_meta.tag_metas, key_id_map)
        objects = VolumeObjectCollection.from_json(data[const.OBJECTS], project_meta, key_id_map)

        planes = {const.AXIAL: None, const.SAGITTAL: None, const.CORONAL: None}
        for plane in data[const.PLANES]:

            plane_name = plane[const.NAME]
            if plane_name in planes.keys():
                if not planes[plane_name]:
                    planes[plane_name] = Plane.from_json(plane, objects, key_id_map)
                else:
                    raise RuntimeError(f'Cannot add more that one plane of type {plane_name}')
            else:
                raise RuntimeError(f'Wrong plane type {plane_name}!')

        return cls(volume_meta=volume_meta,
                   objects=objects,
                   axial=planes[const.AXIAL],
                   sagittal=planes[const.SAGITTAL],
                   coronal=planes[const.CORONAL],
                   tags=tags,
                   description=description,
                   key=volume_key)

    def to_json(self, key_id_map: KeyIdMap = None):
        res_json = {
            const.VOLUME_META: self.volume_meta,
            const.DESCRIPTION: self.description,
            const.KEY: self.key().hex,
            const.TAGS: self.tags.to_json(key_id_map),
            const.OBJECTS: self.objects.to_json(key_id_map),
            const.PLANES: []
        }

        for plane in self.axial, self.sagittal, self.coronal:
            if plane.name in const.PLANE_NAMES:
                res_json[const.PLANES].append(plane.to_json(key_id_map))

        if key_id_map is not None:
            volume_id = key_id_map.get_video_id(self.key())
            if volume_id is not None:
                res_json[const.VOLUME_ID] = volume_id
        return res_json

    def clone(self, volume_meta=None, objects=None, axial=None, sagittal=None, coronal=None, tags=None, description=None):
        return VolumeAnnotation(volume_meta=take_with_default(volume_meta, self.volume_meta),
                                axial=take_with_default(axial, self.axial),
                                coronal=take_with_default(coronal, self.coronal),
                                sagittal=take_with_default(sagittal, self.sagittal),
                                objects=take_with_default(objects, self.objects),
                                tags=take_with_default(tags, self.tags),
                                description=take_with_default(description, self.description))
