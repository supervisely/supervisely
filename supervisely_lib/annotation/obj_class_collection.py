# coding: utf-8

from typing import List
from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection
from supervisely_lib.io.json import JsonSerializable
from supervisely_lib.annotation.obj_class import ObjClass


class ObjClassCollection(KeyIndexedCollection, JsonSerializable):
    """
    Collection that stores ObjClass instances with unique names. It raises error if the name of added item already exists
    """

    item_type = ObjClass

    def to_json(self) -> List[dict]:
        """
        Converts collection to json serializable list. See Supervisely Json format explanation here:
        https://docs.supervise.ly/ann_format/

        Returns:
            json serializable dictionary
        """
        return [obj_class.to_json() for obj_class in self]

    @classmethod
    def from_json(cls, data: List[dict]) -> 'ObjClassCollection':
        """
        Creates collection from json serializable list. See Supervisely Json format explanation here:
        https://docs.supervise.ly/ann_format/

        Returns:
            ObjClassCollection
        """
        obj_classes = [ObjClass.from_json(obj_class_json) for obj_class_json in data]
        return cls(obj_classes)


def make_renamed_classes(src_obj_classes: ObjClassCollection, renamer, skip_missing=False) -> ObjClassCollection:
    renamed_classes = []
    for src_cls in src_obj_classes:
        renamed_name = renamer.rename(src_cls.name)
        if renamed_name is not None:
            renamed_classes.append(src_cls.clone(name=renamed_name))
        elif not skip_missing:
            raise KeyError('Object class name {} could not be mapped to a destination name.'.format(src_cls.name))
    return ObjClassCollection(items=renamed_classes)
