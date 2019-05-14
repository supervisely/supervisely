# coding: utf-8

from supervisely_lib.annotation.tag_meta import TagValueType
from supervisely_lib.collection.key_indexed_collection import KeyObject
from supervisely_lib._utils import take_with_default


class TagJsonFields:
    TAG_NAME = 'name'
    VALUE = 'value'


class Tag(KeyObject):
    def __init__(self, meta, value=None):
        if meta is None:
            raise ValueError('TagMeta is None')
        self._meta = meta
        self._value = value
        if not self._meta.is_valid_value(value):
            raise ValueError('Tag {} can not have value {}'.format(self.meta.name, value))

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
        if self.meta.value_type is TagValueType.NONE:
            return self.meta.name
        else:
            return {
                TagJsonFields.TAG_NAME: self.meta.name,
                TagJsonFields.VALUE: self.value
            }

    @classmethod
    def from_json(cls, data, tag_meta_collection):
        if type(data) is str:
            tag_name = data
            value = None
        else:
            tag_name = data[TagJsonFields.TAG_NAME]
            value = data.get(TagJsonFields.VALUE, None)
        meta = tag_meta_collection.get(tag_name)
        return cls(meta=meta, value=value)

    def get_compact_str(self):
        if (self.meta.value_type != TagValueType.NONE) and (len(str(self.value)) > 0):
            return '{}:{}'.format(self.name, self.value)
        return self.name

    def __eq__(self, other):
        return isinstance(other, Tag) and self.meta == other.meta and self.value == other.value

    def __ne__(self, other):
        return not self == other

    def clone(self, meta=None, value=None):
        return Tag(meta=take_with_default(meta, self.meta), value=take_with_default(value, self.value))

    def __str__(self):
        return '{:<7s}{:<10}{:<7s} {:<13}{:<7s} {:<10}'.format('Name:', self._meta.name,
                                                               'Value type:', self._meta.value_type,
                                                               'Value:', str(self.value))

    @classmethod
    def get_header_ptable(cls):
        return ['Name', 'Value type', 'Value']

    def get_row_ptable(self):
        return [self._meta.name, self._meta.value_type, self.value]