# coding: utf-8

import uuid
from supervisely.annotation.tag import Tag, TagJsonFields
from supervisely._utils import take_with_default
from supervisely.volume_annotation.constants import KEY, ID
from supervisely.video_annotation.key_id_map import KeyIdMap


class VolumeTag(Tag):
    def __init__(
        self,
        meta,
        value=None,
        key=None,
        sly_id=None,
        labeler_login=None,
        updated_at=None,
        created_at=None,
    ):
        super(VolumeTag, self).__init__(
            meta,
            value=value,
            sly_id=sly_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
        self._key = take_with_default(key, uuid.uuid4())

    def key(self):
        return self._key

    def to_json(self, key_id_map: KeyIdMap = None):
        data_json = super(VolumeTag, self).to_json()
        if type(data_json) is str:
            # @TODO: case when tag has no value, super.to_json() returns tag name
            data_json = {TagJsonFields.TAG_NAME: data_json}
        data_json[KEY] = self.key().hex

        if key_id_map is not None:
            item_id = key_id_map.get_tag_id(self.key())
            if item_id is not None:
                data_json[ID] = item_id

        return data_json

    @classmethod
    def from_json(cls, data, tag_meta_collection, key_id_map: KeyIdMap = None):
        temp = super(VolumeTag, cls).from_json(data, tag_meta_collection)
        key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_tag(key, data.get(ID, None))

        return cls(
            meta=temp.meta,
            value=temp.value,
            key=key,
            sly_id=temp.sly_id,
            labeler_login=temp.labeler_login,
            updated_at=temp.updated_at,
            created_at=temp.created_at,
        )

    def clone(
        self,
        meta=None,
        value=None,
        key=None,
        sly_id=None,
        labeler_login=None,
        updated_at=None,
        created_at=None,
    ):
        return VolumeTag(
            meta=take_with_default(meta, self.meta),
            value=take_with_default(value, self.value),
            key=take_with_default(key, self.key),
            sly_id=take_with_default(sly_id, self.sly_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at),
        )
