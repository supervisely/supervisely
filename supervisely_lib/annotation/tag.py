# coding: utf-8

from supervisely_lib.annotation.tag_meta import TagValueType
from supervisely_lib.collection.key_indexed_collection import KeyObject
from supervisely_lib._utils import take_with_default


class TagJsonFields:
    TAG_NAME = 'name'
    VALUE = 'value'
    LABELER_LOGIN = 'labelerLogin'
    UPDATED_AT = 'updatedAt'
    CREATED_AT = 'createdAt'
    ID = 'id'
    #TAG_META_ID = 'tagId'


class Tag(KeyObject):
    '''
    This is a class for creating and using Tags objects. The tags can be attached both to whole images and to
    individual geometric labels.
    '''
    def __init__(self, meta, value=None, sly_id=None, labeler_login=None, updated_at=None, created_at=None):
        '''
        The constructor for Tag class.
        :param meta: Tag metadata: it include tag name, value type, and possible values for tags with enum values.
        When creating a new tag, the value is automatically cross-checked against the metadata to make sure the value
        is valid.
        :param value: There are 3 possible value types of value: ANY_NUMBER for numeric values,
        ANY_STRING for arbitrary string values, ONEOF_STRING for string values restricted to a given whitelist/
        '''

        if meta is None:
            raise ValueError('TagMeta is None')
        self._meta = meta
        self._value = value
        if not self._meta.is_valid_value(value):
            raise ValueError('Tag {} can not have value {}'.format(self.meta.name, value))
        self.labeler_login = labeler_login
        self.updated_at = updated_at
        self.created_at = created_at
        self.sly_id = sly_id

    @property
    def meta(self):
        return self._meta

    @property
    def value(self):
        return self._value

    @property
    def name(self):
        return self._meta.name

    def key(self):
        return self._meta.key()

    def to_json(self):
        '''
        The function to_json convert tag to json format
        :return: tag in json format
        '''
        res = {
            TagJsonFields.TAG_NAME: self.meta.name,
            #TagJsonFields.VALUE: self.value
        }
        if self.meta.value_type != TagValueType.NONE:
            res[TagJsonFields.VALUE] = self.value
        if self.labeler_login is not None:
            res[TagJsonFields.LABELER_LOGIN] = self.labeler_login
        if self.updated_at is not None:
            res[TagJsonFields.UPDATED_AT] = self.updated_at
        if self.created_at is not None:
            res[TagJsonFields.CREATED_AT] = self.created_at
        # if self.meta.value_type == TagValueType.NONE:
        #     return self.meta.name
        # else:
        #     return {
        #         TagJsonFields.TAG_NAME: self.meta.name,
        #         TagJsonFields.VALUE: self.value
        #     }
        return res

    @classmethod
    def from_json(cls, data, tag_meta_collection):
        '''
        The function from_json convert tag from json format to Tag class object.
        :param data: input tag in json format
        :param tag_meta_collection: TagCollection class object
        :return: Tag class object
        '''
        if type(data) is str:
            tag_name = data
            value = None
            labeler_login = None
            updated_at = None
            created_at = None
            sly_id = None
        else:
            tag_name = data[TagJsonFields.TAG_NAME]
            value = data.get(TagJsonFields.VALUE, None)
            labeler_login = data.get(TagJsonFields.LABELER_LOGIN, None)
            updated_at = data.get(TagJsonFields.UPDATED_AT, None)
            created_at = data.get(TagJsonFields.CREATED_AT, None)
            sly_id = data.get(TagJsonFields.ID, None)
        meta = tag_meta_collection.get(tag_name)
        return cls(meta=meta, value=value, sly_id=sly_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

    def get_compact_str(self):
        '''
        The function get_compact_str displays information about tag(name, value) in string format
        '''
        if (self.meta.value_type != TagValueType.NONE) and (len(str(self.value)) > 0):
            return '{}:{}'.format(self.name, self.value)
        return self.name

    def __eq__(self, other):
        return isinstance(other, Tag) and self.meta == other.meta and self.value == other.value

    def __ne__(self, other):
        return not self == other

    def clone(self, meta=None, value=None, sly_id=None, labeler_login=None, updated_at=None, created_at=None):
        '''
        The function clone make copy of the Tag class object
        :return: Tag class object
        '''
        return Tag(meta=take_with_default(meta, self.meta),
                   value=take_with_default(value, self.value),
                   sly_id=take_with_default(sly_id, self.sly_id),
                   labeler_login=take_with_default(labeler_login, self.labeler_login),
                   updated_at=take_with_default(updated_at, self.updated_at),
                   created_at=take_with_default(created_at, self.created_at))

    def __str__(self):
        return '{:<7s}{:<10}{:<7s} {:<13}{:<7s} {:<10}'.format('Name:', self._meta.name,
                                                               'Value type:', self._meta.value_type,
                                                               'Value:', str(self.value))

    @classmethod
    def get_header_ptable(cls):
        return ['Name', 'Value type', 'Value']

    def get_row_ptable(self):
        '''
        :return: information about Tag class object(name of meta, value of the tag, and value type)
        '''
        return [self._meta.name, self._meta.value_type, self.value]