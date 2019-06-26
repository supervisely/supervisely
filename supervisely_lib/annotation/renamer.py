# coding: utf-8
from copy import copy


MATCH_ALL = '__all__'


def is_name_included(name, enabled_names):
    return (enabled_names == MATCH_ALL) or (type(enabled_names) != str and name in enabled_names)


class Renamer:
    ADD_SUFFIX = 'add_suffix'
    SAVE_CLASSES = 'save_classes'  # Deprecated. Use SAVE_NAMES in new code.
    SAVE_NAMES = 'save_names'      # New field with more generic name.

    def __init__(self, add_suffix='', save_names=None):
        self._add_suffix = add_suffix
        self._save_names = copy(save_names) if save_names is not None else MATCH_ALL

    def rename(self, name):
        return (name + self._add_suffix) if is_name_included(name, self._save_names) else None

    def to_json(self):
        return {Renamer.ADD_SUFFIX: self._add_suffix, Renamer.SAVE_CLASSES: self._save_names}

    @staticmethod
    def from_json(renamer_json):
        enabled_names = renamer_json.get(Renamer.SAVE_NAMES)
        if enabled_names is None:
            enabled_names = renamer_json.get(Renamer.SAVE_CLASSES)
        add_suffix = renamer_json.get(Renamer.ADD_SUFFIX, '')
        return Renamer(add_suffix=add_suffix, save_names=enabled_names)
