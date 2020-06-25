# coding: utf-8

import uuid
from supervisely_lib.annotation.tag import Tag, TagJsonFields
from supervisely_lib._utils import take_with_default
from supervisely_lib.video_annotation.constants import KEY, ID, FRAME_RANGE
from supervisely_lib.video_annotation.key_id_map import KeyIdMap


class VideoTag(Tag):
    '''
    This is a class for creating and using VideoTag objects for videos
    '''
    def __init__(self, meta, value=None, frame_range=None, key=None, sly_id=None, labeler_login=None, updated_at=None, created_at=None):
        '''
        :param meta: Tag metadata: it include tag name, value type, and possible values for tags with enum values.
        When creating a new tag, the value is automatically cross-checked against the metadata to make sure the value
        is valid.
        :param value: There are 3 possible value types of value: ANY_NUMBER for numeric values,
        ANY_STRING for arbitrary string values, ONEOF_STRING for string values restricted to a given whitelist
        :param frame_range: tuple or list of integers
        :param key: uuid class object
        '''
        super(VideoTag, self).__init__(meta, value=value, sly_id=sly_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)
        
        self._frame_range = None
        if frame_range is not None:
            if not isinstance(frame_range, (tuple, list)):
                raise TypeError('{!r} has to be a tuple or a list. Given type "{}".'.format(FRAME_RANGE, type(frame_range)))
            else:
                self._frame_range = list(frame_range)

        self._key = take_with_default(key, uuid.uuid4())

    @property
    def frame_range(self):
        return self._frame_range

    def key(self):
        return self._key

    def to_json(self, key_id_map: KeyIdMap = None):
        '''
        The function to_json convert VideoTag class object to json format
        :param key_id_map: KeyIdMap class object
        :return: VideoTag in json format
        '''
        data_json = super(VideoTag, self).to_json()
        if type(data_json) is str:
            # @TODO: case when tag has no value, super.to_json() returns tag name
            data_json = {TagJsonFields.TAG_NAME: data_json}
        if self.frame_range is not None:
            data_json[FRAME_RANGE] = self.frame_range
        data_json[KEY] = self.key().hex

        if key_id_map is not None:
            item_id = key_id_map.get_tag_id(self.key())
            if item_id is not None:
                data_json[ID] = item_id

        return data_json

    @classmethod
    def from_json(cls, data, tag_meta_collection, key_id_map: KeyIdMap = None):
        '''
        The function from_json convert VideoTag from json format to VideoTag class object.
        :param data: input VideoTag in json format
        :param tag_meta_collection: VideoTagCollection
        :param key_id_map: KeyIdMap class object
        :return: VideoTag class object
        '''
        temp = super(VideoTag, cls).from_json(data, tag_meta_collection)
        frame_range = data.get(FRAME_RANGE, None)
        key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_tag(key, data.get(ID, None))

        return cls(meta=temp.meta, value=temp.value, frame_range=frame_range, key=key,
                   sly_id=temp.sly_id, labeler_login=temp.labeler_login, updated_at=temp.updated_at, created_at=temp.created_at)

    def get_compact_str(self):
        '''
        :return: string with information about tag(name, value) and range of frames
        '''
        res = super(VideoTag, self).get_compact_str()
        if self.frame_range is not None:
            res = "{}[{} - {}]".format(res, self.frame_range[0], self.frame_range[1])
        return res

    def __eq__(self, other):
        return isinstance(other, VideoTag) and \
               self.meta == other.meta and \
               self.value == other.value and \
               self.frame_range == other.frame_range

    def clone(self, meta=None, value=None, frame_range=None, key=None,
                    sly_id=None, labeler_login=None, updated_at=None, created_at=None):
        '''
        :param meta: Tag metadata
        :param value: There are 3 possible value types of value: ANY_NUMBER for numeric values,
        ANY_STRING for arbitrary string values, ONEOF_STRING for string values restricted to a given whitelist
        :param frame_range: tuple or list of integers
        :param key: uuid class object
        :return: VideoTag class object
        '''
        return VideoTag(meta=take_with_default(meta, self.meta),
                        value=take_with_default(value, self.value),
                        frame_range=take_with_default(frame_range, self.frame_range),
                        key=take_with_default(key, self.key),
                        sly_id=take_with_default(sly_id, self.sly_id),
                        labeler_login=take_with_default(labeler_login, self.labeler_login),
                        updated_at=take_with_default(updated_at, self.updated_at),
                        created_at=take_with_default(created_at, self.created_at))

    def __str__(self):
        return '{:<7s}{:<10}{:<7s} {:<13}{:<7s} {:<10} {:<12}'.format('Name:', self._meta.name,
                                                               'Value type:', self._meta.value_type,
                                                               'Value:', str(self.value),
                                                               'FrameRange', str(self.frame_range))

    @classmethod
    def get_header_ptable(cls):
        return ['Name', 'Value type', 'Value', 'Frame range']

    def get_row_ptable(self):
        return [self._meta.name, self._meta.value_type, self.value, self.frame_range]
