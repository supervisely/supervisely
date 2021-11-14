import uuid
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
                 description: str = None,
                 key=None):
        pass

    @classmethod
    def from_json(cls, data, project_meta, key_id_map: KeyIdMap=None):
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



