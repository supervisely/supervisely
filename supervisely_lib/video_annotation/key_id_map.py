# coding: utf-8

import uuid
from bidict import bidict
from supervisely_lib.io.json import dump_json_file, load_json_file

TAGS = 'tags'
OBJECTS = 'objects'
FIGURES = 'figures'
VIDEOS = 'videos'

ALLOWED_KEY_TYPES = [TAGS, OBJECTS, VIDEOS, FIGURES]


class KeyIdMap:
    '''
    This is a class for creating and using key_id_maps for videos
    '''
    def __init__(self):
        self._data = dict()
        self._data[TAGS] = bidict()
        self._data[OBJECTS] = bidict()
        self._data[FIGURES] = bidict()
        self._data[VIDEOS] = bidict()

    def _add(self, key_type, key: uuid.UUID, id: int = None):
        '''
        Add given data in self._data dictionary. Raise error if data type of any parameter is invalid
        :param key_type: str
        :param key: uuid class object
        :param id: int
        '''
        if key_type not in ALLOWED_KEY_TYPES:
            raise RuntimeError("Key type {!r} is not allowed. Allowed types are {}".format(key_type, ALLOWED_KEY_TYPES))
        if type(key) is not uuid.UUID:
            raise RuntimeError("Key should be of type uuid.UUID")
        if id is not None and type(id) is not int:
            raise RuntimeError("Id should be of type int")
        self._data[key_type].update(bidict({key: id}))

    def add_object(self, key: uuid.UUID, id: int):
        '''
        Add objects in self._data dictionary with given key and id.
        :param key: uuid class object
        :param id: int
        '''
        self._add(OBJECTS, key, id)

    def add_tag(self, key: uuid.UUID, id: int):
        '''
        Add tags in self._data dictionary with given key and id.
        :param key: uuid class object
        :param id: int
        '''
        self._add(TAGS, key, id)

    def add_figure(self, key: uuid.UUID, id: int):
        '''
        Add figures in self._data dictionary with given key and id.
        :param key: uuid class object
        :param id: int
        '''
        self._add(FIGURES, key, id)

    def add_video(self, key: uuid.UUID, id: int):
        '''
        Add videos in self._data dictionary with given key and id.
        :param key: uuid class object
        :param id: int
        '''
        self._add(VIDEOS, key, id)

    def _get_id_by_key(self, key_type, key: uuid.UUID):
        '''
        :param key_type: str
        :param key: uuid class object
        :return: Id by given key. None if there is no such key. Raise error if key type is not uuid.UUID
        '''
        if type(key) is not uuid.UUID:
            raise RuntimeError("Key should be of type uuid.UUID")

        if key in self._data[key_type]:
            return self._data[key_type][key]
        else:
            return None

    def _get_key_by_id(self, key_type, id: int):
        '''
        :param key_type: str
        :param id: int
        :return: Key by given id. None if there is no such id. Raise error if id type is not int
        '''
        if type(id) is not int:
            raise RuntimeError("Id should be of type int")
        if id not in self._data[key_type].inverse:
            return None
        return self._data[key_type].inverse[id]

    def get_object_id(self, key: uuid.UUID):
        '''
        :param key: uuid class object
        :return: Id by given key for objects field.
        '''
        return self._get_id_by_key(OBJECTS, key)

    def get_tag_id(self, key: uuid.UUID):
        '''
        :param key: uuid class object
        :return: Id by given key for tags field.
        '''
        return self._get_id_by_key(TAGS, key)

    def get_figure_id(self, key: uuid.UUID):
        '''
        :param key: uuid class object
        :return: Id by given key for figures field.
        '''
        return self._get_id_by_key(FIGURES, key)

    def get_video_id(self, key: uuid.UUID):
        '''
        :param key: uuid class object
        :return: Id by given key for videos field.
        '''
        return self._get_id_by_key(VIDEOS, key)

    def get_object_key(self, id: int):
        '''
        :param id: int
        :return: Key by given id for objects field.
        '''
        return self._get_key_by_id(OBJECTS, id)

    def get_tag_key(self, id: int):
        '''
        :param id: int
        :return: Key by given id for tags field.
        '''
        return self._get_key_by_id(TAGS, id)

    def get_figure_key(self, id: int):
        '''
        :param id: int
        :return: Key by given id for figures field.
        '''
        return self._get_key_by_id(FIGURES, id)

    def get_video_key(self, id: int):
        '''
        :param id: int
        :return: Key by given id for videos field.
        '''
        return self._get_key_by_id(VIDEOS, id)

    def to_dict(self):
        '''
        Convert current self._data dictionary with bidict values to dictionary with dict values
        :return: dict
        '''
        simple_dict = {}
        for type_str, value_bidict in self._data.items():
            sub_dict = {}
            for uuid, int_id in value_bidict.items():
                sub_dict[uuid.hex] = int_id
            simple_dict[type_str] = sub_dict
        return simple_dict

    def dump_json(self, path):
        '''
        Save current class object data in json format by given path
        :param path: str
        '''
        simple_dict = self.to_dict()
        dump_json_file(simple_dict, path, indent=4)

    @classmethod
    def load_json(cls, path):
        '''
        Download json data by given path and convert in to dict with bidict values
        :param path: str
        :return: dict
        '''
        simple_dict = load_json_file(path)
        result = cls()
        for key_type, value_dict in simple_dict.items():
            for key_str, id in value_dict.items():
                result._add(key_type, uuid.UUID(key_str), id)
        return result

    @classmethod
    def _add_to(cls, key_id_map, key_type, keys, ids):
        '''
        Add given values(keys, ids) to KeyIdMap class object with given type of key
        :param key_id_map: KeyIdMap class object
        :param key_type: str
        :param keys: list of uuid class objects
        :param ids: list of integers
        :return: None if key_id_map parameter is None
        '''
        if key_id_map is None:
            return
        for key, id in zip(keys, ids):
            key_id_map._add(key_type, key, id)

    @classmethod
    def add_tags_to(cls, key_id_map, keys, ids):
        '''
        Add tags type of key with given values(keys, ids) to KeyIdMap class object
        :param key_id_map: KeyIdMap class object
        :param keys: list of uuid class objects
        :param ids: list of integers
        '''
        cls._add_to(key_id_map, TAGS, keys, ids)

    @classmethod
    def add_tag_to(cls, key_id_map, key, id):
        '''
        Add tags type of key with given key and id to KeyIdMap class object
        :param key_id_map: KeyIdMap class object
        :param key: uuid class object
        :param id: int
        '''
        cls._add_tags_to(key_id_map, [key], [id])

    @classmethod
    def add_objects_to(cls, key_id_map, keys, ids):
        '''
        Add objects type of key with given values(keys, ids) to KeyIdMap class object
        :param key_id_map: KeyIdMap class object
        :param keys: list of uuid class objects
        :param ids: list of integers
        '''
        cls._add_to(key_id_map, OBJECTS, keys, ids)

    @classmethod
    def add_object_to(cls, key_id_map, key, id):
        '''
        Add objects type of key with given key and id to KeyIdMap class object
        :param key_id_map: KeyIdMap class object
        :param key: uuid class object
        :param id: int
        '''
        cls._add_objects_to(key_id_map, [key], [id])

    @classmethod
    def add_figures_to(cls, key_id_map, keys, ids):
        '''
        Add figures type of key with given values(keys, ids) to KeyIdMap class object
        :param key_id_map: KeyIdMap class object
        :param keys: list of uuid class objects
        :param ids: list of integers
        '''
        cls._add_to(key_id_map, FIGURES, keys, ids)

    @classmethod
    def add_figure_to(cls, key_id_map, key, id):
        '''
        Add figure type of key with given key and id to KeyIdMap class object
        :param key_id_map: KeyIdMap class object
        :param key: uuid class object
        :param id: int
        '''
        cls._add_figures_to(key_id_map, [key], [id])

    @classmethod
    def add_videos_to(cls, key_id_map, keys, ids):
        '''
        Add videos type of key with given values(keys, ids) to KeyIdMap class object
        :param key_id_map: KeyIdMap class object
        :param keys: list of uuid class objects
        :param ids: list of integers
        '''
        cls._add_to(key_id_map, VIDEOS, keys, ids)

    @classmethod
    def add_video_to(cls, key_id_map, key, id):
        '''
        Add video type of key with given key and id to KeyIdMap class object
        :param key_id_map: KeyIdMap class object
        :param key: uuid class object
        :param id: int
        '''
        cls._add_videos_to(key_id_map, [key], [id])
