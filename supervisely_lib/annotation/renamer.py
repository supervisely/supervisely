# coding: utf-8
from copy import copy


MATCH_ALL = '__all__'


def is_name_included(name, enabled_names):
    return (enabled_names == MATCH_ALL) or (type(enabled_names) != str and name in enabled_names)


class Renamer:
    ADD_SUFFIX = 'add_suffix'
    SAVE_CLASSES = 'save_classes'

    def __init__(self, add_suffix='', enabled_classes=None):
        self._add_suffix = add_suffix
        self._enabled_classes = copy(enabled_classes) or MATCH_ALL

    def rename(self, name):
        return (name + self._add_suffix) if is_name_included(name, self._enabled_classes) else None

    def to_json(self):
        return {Renamer.ADD_SUFFIX: self._add_suffix, Renamer.SAVE_CLASSES: self._enabled_classes}

    @staticmethod
    def from_json(renamer_json):
        return Renamer(add_suffix=renamer_json[Renamer.ADD_SUFFIX], enabled_classes=renamer_json[Renamer.SAVE_CLASSES])
