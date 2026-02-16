# coding: utf-8
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.annotation.renamer import Renamer


class ObjClassMapper:
    """
    """
    def map(self, src: ObjClass) -> ObjClass:
        """
        """
        raise NotImplementedError()


class RenamingObjClassMapper(ObjClassMapper):
    """
    This is a class for renaming ObjClass in given ObjClassCollection
    """

    def __init__(self, dest_obj_classes: ObjClassCollection, renamer: Renamer):
        """
        :param dest_obj_classes: :class:`~supervisely.annotation.obj_class_collection.ObjClassCollection` object to map to.
        :type dest_obj_classes: :class:`~supervisely.annotation.obj_class_collection.ObjClassCollection`
        :param renamer: :class:`~supervisely.annotation.renamer.Renamer` object to use for renaming.
        :type renamer: :class:`~supervisely.annotation.renamer.Renamer`
        """
        self._dest_obj_classes = dest_obj_classes
        self._renamer = renamer

    def map(self, src: ObjClass) -> ObjClass:
        """
        The function map rename ObjClass in given collection
        :returns: ObjClass object
        :rtype: :class:`~supervisely.annotation.obj_class.ObjClass`
        """
        dest_name = self._renamer.rename(src.name)
        return self._dest_obj_classes.get(dest_name, None) if (dest_name is not None) else None
