# coding: utf-8
from supervisely_lib.annotation.obj_class import ObjClass
from supervisely_lib.annotation.obj_class_collection import ObjClassCollection
from supervisely_lib.annotation.renamer import Renamer


class ObjClassMapper:
    def map(self, src: ObjClass) -> ObjClass:
        raise NotImplementedError()


class RenamingObjClassMapper(ObjClassMapper):
    def __init__(self, dest_obj_classes: ObjClassCollection, renamer: Renamer):
        self._dest_obj_classes = dest_obj_classes
        self._renamer = renamer

    def map(self, src: ObjClass) -> ObjClass:
        dest_name = self._renamer.rename(src.name)
        return self._dest_obj_classes.get(dest_name, None) if (dest_name is not None) else None


def replace_labels_classes(labels, obj_class_mapper: ObjClassMapper, skip_missing=False) -> list:
    result = []
    for label in labels:
        dest_obj_class = obj_class_mapper.map(label.obj_class)
        if dest_obj_class is not None:
            result.append(label.clone(obj_class=dest_obj_class))
        elif not skip_missing:
            raise KeyError(
                'Object class {} could not be mapped to a destination object class.'.format(label.obj_class.name))
    return result
