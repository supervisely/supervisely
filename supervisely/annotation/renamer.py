# coding: utf-8
from copy import copy
from typing import List, Optional

MATCH_ALL = '__all__'


def is_name_included(name: str, enabled_names: List[str]) -> bool:
    """"""
    return (enabled_names == MATCH_ALL) or (type(enabled_names) != str and name in enabled_names)


class Renamer:
    """
    Rule-based name transformer used by mappers when cloning object classes.
    """

    ADD_SUFFIX = 'add_suffix'
    """"""
    SAVE_CLASSES = 'save_classes'  # Deprecated. Use SAVE_NAMES in new code.
    """"""
    SAVE_NAMES = 'save_names'  # New field with more generic name.
    """"""

    def __init__(self, add_suffix: str = "", save_names: Optional[List[str]] = None):
        """
        :param add_suffix: Suffix to be added to the processed object names.
        :type add_suffix: str
        :param save_names: List of the object names to which you will need to add a suffix.
        :type save_names: list
        """
        self._add_suffix = add_suffix
        self._save_names = copy(save_names) if save_names is not None else MATCH_ALL

    def rename(self, name):
        """
        The function add special suffix to input name
        :param name: name to be changed
        :returns: new name
        """
        return (name + self._add_suffix) if is_name_included(name, self._save_names) else None

    def to_json(self):
        """
        The function to_json convert Renamer to json format
        :returns: Renamer in json format
        :rtype: dict
        """
        return {Renamer.ADD_SUFFIX: self._add_suffix, Renamer.SAVE_CLASSES: self._save_names}

    @staticmethod
    def from_json(renamer_json):
        """
        The function from_json convert Renamer from json format to Renamer class object.
        :param renamer_json: Renamer in json format
        :returns: Renamer object
        :rtype: :class:`~supervisely.annotation.renamer.Renamer`
        """
        enabled_names = renamer_json.get(Renamer.SAVE_NAMES)
        if enabled_names is None:
            enabled_names = renamer_json.get(Renamer.SAVE_CLASSES)
        add_suffix = renamer_json.get(Renamer.ADD_SUFFIX, '')
        return Renamer(add_suffix=add_suffix, save_names=enabled_names)
