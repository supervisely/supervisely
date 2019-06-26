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
