# coding: utf-8

from typing import List
from copy import deepcopy
from supervisely_lib.imaging.color import random_rgb, rgb2hex, hex2rgb, _validate_color
from supervisely_lib.io.json import JsonSerializable
from supervisely_lib.collection.key_indexed_collection import KeyObject
from supervisely_lib._utils import take_with_default


class TagValueType:
    NONE = 'none'
    ANY_NUMBER = 'any_number'
    ANY_STRING = 'any_string'
    ONEOF_STRING = 'oneof_string'


class TagMetaJsonFields:
    ID = 'id'
    NAME = 'name'
    VALUE_TYPE = 'value_type'
    VALUES = 'values'
    COLOR = 'color'
    APPLICABLE_TYPE = 'applicable_type'
    HOTKEY = "hotkey"
    APPLICABLE_CLASSES = 'classes'


class TagApplicableTo:
    ALL = 'all' # both images and objects
    IMAGES_ONLY = 'imagesOnly'
    OBJECTS_ONLY = 'objectsOnly'


SUPPORTED_TAG_VALUE_TYPES = [TagValueType.NONE, TagValueType.ANY_NUMBER, TagValueType.ANY_STRING, TagValueType.ONEOF_STRING]
SUPPORTED_APPLICABLE_TO = [TagApplicableTo.ALL, TagApplicableTo.IMAGES_ONLY, TagApplicableTo.OBJECTS_ONLY]


class TagMeta(KeyObject, JsonSerializable):
    '''
    This is a class for creating and using TagMeta objects. It include tag name, value type, and possible values for
    tags with enum values.
    '''
    def __init__(self, name: str, value_type: str, possible_values: List[str] = None, color: List[int]=None, sly_id=None,
                 hotkey: str = None, applicable_to: str = None, applicable_classes: List[str]=None):
        """
        :param name: str
        :param value_type: str (one of TagValueType fields)
        :param values: list of possible values (i.e. [str]) or None
        :param color: [R, G, B]
        """

        if value_type not in SUPPORTED_TAG_VALUE_TYPES:
            raise ValueError("value_type = {!r} is unknown, should be one of {}"
                             .format(value_type, SUPPORTED_TAG_VALUE_TYPES))

        self._name = name
        self._value_type = value_type
        self._possible_values = possible_values
        self._color = random_rgb() if color is None else deepcopy(color)
        self._sly_id = sly_id
        self._hotkey = take_with_default(hotkey, "")
        self._applicable_to = take_with_default(applicable_to, TagApplicableTo.ALL)
        self._applicable_classes = take_with_default(applicable_classes, [])
        if self._applicable_to not in SUPPORTED_APPLICABLE_TO:
            raise ValueError("applicable_to = {!r} is unknown, should be one of {}"
                             .format(self._applicable_to, SUPPORTED_APPLICABLE_TO))

        if self._value_type == TagValueType.ONEOF_STRING:
            if self._possible_values is None:
                raise ValueError("TagValueType is ONEOF_STRING. List of possible values have to be defined.")
            if not all(isinstance(item, str) for item in self._possible_values):
                raise ValueError("TagValueType is ONEOF_STRING. All possible values have to be strings")
        elif self._possible_values is not None:
            raise ValueError("TagValueType is {!r}. possible_values variable have to be None".format(self._value_type))

        _validate_color(self._color)

    @property
    def name(self):
        return self._name

    def key(self):
        return self.name

    @property
    def value_type(self):
        return self._value_type

    @property
    def possible_values(self):
        return self._possible_values.copy() if self._possible_values is not None else None

    @property
    def color(self):
        return self._color.copy()

    @property
    def sly_id(self):
        return self._sly_id

    @property
    def hotkey(self):
        return self._hotkey

    @property
    def applicable_to(self):
        return self._applicable_to

    @property
    def applicable_classes(self):
        return self._applicable_classes

    def to_json(self):
        '''
        The function to_json convert TagMeta object to json format
        :return: tagmeta in json format
        '''
        jdict = {
            TagMetaJsonFields.NAME: self.name,
            TagMetaJsonFields.VALUE_TYPE: self.value_type,
            TagMetaJsonFields.COLOR: rgb2hex(self.color)
        }
        if self.value_type == TagValueType.ONEOF_STRING:
            jdict[TagMetaJsonFields.VALUES] = self.possible_values

        if self.sly_id is not None:
            jdict[TagMetaJsonFields.ID] = self.sly_id
        if self._hotkey is not None:
            jdict[TagMetaJsonFields.HOTKEY] = self.hotkey
        if self._applicable_to is not None:
            jdict[TagMetaJsonFields.APPLICABLE_TYPE] = self.applicable_to
        if self._applicable_classes is not None:
            jdict[TagMetaJsonFields.APPLICABLE_CLASSES] = self.applicable_classes

        return jdict

    @classmethod
    def from_json(cls, data):
        '''
        The function from_json convert tagmeta from json format to TagMeta class object.
        :param data: input tagmeta in json format
        :return: TagMeta class object
        '''
        if isinstance(data, str):
            return cls(name=data, value_type=TagValueType.NONE)
        elif isinstance(data, dict):
            name = data[TagMetaJsonFields.NAME]
            value_type = data[TagMetaJsonFields.VALUE_TYPE]
            values = data.get(TagMetaJsonFields.VALUES)
            color = data.get(TagMetaJsonFields.COLOR)
            if color is not None:
                color = hex2rgb(color)
            sly_id = data.get(TagMetaJsonFields.ID, None)

            hotkey = data.get(TagMetaJsonFields.HOTKEY, "")
            applicable_to = data.get(TagMetaJsonFields.APPLICABLE_TYPE, TagApplicableTo.ALL)
            applicable_classes = data.get(TagMetaJsonFields.APPLICABLE_CLASSES, [])

            return cls(name=name, value_type=value_type, possible_values=values, color=color, sly_id=sly_id,
                       hotkey=hotkey, applicable_to=applicable_to, applicable_classes=applicable_classes)
        else:
            raise ValueError('Tags must be dict or str types.')

    def add_possible_value(self, value):
        '''
        The function add_possible_value add new value to the list of possible_values. If value_type is not 'oneof_string'
        it generate ValueError error.
        :param value: value added
        :return: TagMeta class object
        '''
        if self.value_type is TagValueType.ONEOF_STRING:
            if value in self._possible_values:
                raise ValueError('Value {} already exists for tag {}'.format(value, self.name))
            else:
                return self.clone(possible_values=[*self.possible_values, value])
        else:
            raise ValueError("Tag {!r} has type {!r}. Possible value can be added only to oneof_string".format(self.name, self.value_type))

    def is_valid_value(self, value):
        '''
        The function is_valid_value cross-checked value against the value_type to make sure the value is valid.
        If value is unsupported it generate ValueError error.
        :return: True if value is supported, False in other way.
        '''
        if self.value_type == TagValueType.NONE:
            return value is None
        elif self.value_type == TagValueType.ANY_NUMBER:
            return isinstance(value, (int, float))
        elif self.value_type == TagValueType.ANY_STRING:
            return isinstance(value, str)
        elif self.value_type == TagValueType.ONEOF_STRING:
            return isinstance(value, str) and (value in self._possible_values)
        else:
            raise ValueError('Unsupported TagValueType detected ({})!'.format(self.value_type))

    def __eq__(self, other):
        # TODO compare colors also here (need to check the usages and replace with is_compatible() where appropriate).
        return (isinstance(other, TagMeta) and
                self.name == other.name and
                self.value_type == other.value_type and
                self.possible_values == other.possible_values)

    def __ne__(self, other):
        return not self == other

    def is_compatible(self, other):
        '''
        The function is_compatible checks the input data against the given TagMeta object
        :param other: some Object
        '''
        return (isinstance(other, TagMeta) and
                self.name == other.name and
                self.value_type == other.value_type and
                self.possible_values == other.possible_values)

    def clone(self, name=None, value_type=None, possible_values=None, color=None, sly_id=None,
              hotkey=None, applicable_to=None, applicable_classes=None):
        '''
        The function clone make copy of the TagMeta class object
        :return: TagMeta class object
        '''
        return TagMeta(name=take_with_default(name, self.name),
                       value_type=take_with_default(value_type, self.value_type),
                       possible_values=take_with_default(possible_values, self.possible_values),
                       color=take_with_default(color, self.color),
                       sly_id=take_with_default(sly_id, self.sly_id),
                       hotkey=take_with_default(hotkey, self.hotkey),
                       applicable_to=take_with_default(applicable_to, self.applicable_to),
                       applicable_classes=take_with_default(applicable_classes, self.applicable_classes))

    def __str__(self):
        return "{:<7s}{:<24} {:<7s}{:<13} {:<13s}{:<10} {:<13s}{:<10} {:<13s}{:<10} {:<13s}{}".format(
            'Name:', self.name, 'Value type:', self.value_type, 'Possible values:', str(self.possible_values),
            'Hotkey', self.hotkey, 'Applicable to', self.applicable_to, 'Applicable classes', self.applicable_classes)

    @classmethod
    def get_header_ptable(cls):
        return ['Name', 'Value type', 'Possible values', 'Hotkey', 'Applicable to', 'Applicable classes']

    def get_row_ptable(self):
        '''
        :return: information about TagMeta class object(name of meta, type value, and list of possible values)
        '''
        return [self.name, self.value_type, self.possible_values, self.hotkey, self.applicable_to, self.applicable_classes]