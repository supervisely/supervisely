# coding: utf-8

from copy import deepcopy
from enum import Enum

from .color_utils import gen_new_color


class FigShape(Enum):
    POINT = 1
    RECTANGLE = 2
    LINE = 3
    POLYGON = 4
    BITMAP = 5


# storage for list of classes, preserving order
class FigClasses(object):
    allowed_shapes = set((x.name.lower() for x in FigShape))

    def __init__(self, classes_lst=None):
        # use both dict and list for classes to preserve source order of classes
        if classes_lst is None:
            self._classes_lst = []
        else:
            self._classes_lst = deepcopy(classes_lst)
            for cls_dct in classes_lst:
                self._check_input_class_dct(cls_dct)
                self._check_add_color(cls_dct)

        self._classes_dct = {x['title']: x for x in self._classes_lst}
        if len(self._classes_dct) != len(self._classes_lst):
            raise RuntimeError('Non-unique class names are not allowed.')

    # class dictionary by title; None if not found
    def __getitem__(self, item):
        return self._classes_dct.get(item, None)

    def __iter__(self):
        return self._classes_lst.__iter__()

    def __len__(self):
        return len(self._classes_lst)

    @classmethod
    def _check_input_class_dct(cls, dct):
        in_title = dct.get('title', None)
        if type(in_title) is not str:
            raise RuntimeError('Unique string title is required.')
        shape = dct.get('shape', None)
        if shape not in cls.allowed_shapes:
            raise RuntimeError('Shape not allowed: {}.'.format(shape))

    @classmethod
    def _check_add_color(cls, dct):
        if 'color' not in dct:
            dct['color'] = gen_new_color()
        else:
            pass  # @TODO: validate string

    # 'raw' view; use read-only pls
    @property
    def py_container(self):
        return self._classes_lst

    @property
    def unique_names(self):
        return set(self._classes_dct.keys())

    # require same shape if class with same title already exists
    # will not update color or other fields for existing class
    def add(self, new_class_dct):
        self._check_input_class_dct(new_class_dct)
        new_title = new_class_dct['title']
        new_shape = new_class_dct['shape']

        existing_class = self._classes_dct.get(new_title, None)
        if existing_class is not None:
            existing_shape = existing_class['shape']
            if new_shape != existing_shape:
                raise RuntimeError(
                    'Trying to add new class ({}) with shape ({}). Same class with different shape ({}) exists.'.format(
                        new_title, new_shape, existing_shape
                    ))
        else:
            new_dct = deepcopy(new_class_dct)
            self._check_add_color(new_dct)
            self._classes_lst.append(new_dct)
            self._classes_dct[new_title] = new_dct
            # allow other (non-required) fields for classes

    def delete(self, title):
        cls_to_rm = self[title]
        if cls_to_rm is None:
            raise RuntimeError("Can not delete class {}. Not found".format(title))
        self._classes_lst.remove(cls_to_rm)
        del self._classes_dct[title]

    def replace(self, old_title, new_class_dct):
        if new_class_dct['title'] != old_title:
            self.delete(old_title)
        self.add(new_class_dct)

    def rename(self, old_title, new_title):
        new_cls = self[old_title]
        if new_cls is None:
            raise RuntimeError("Can not rename class {}. Not found".format(old_title))
        new_cls['title'] = new_title
        self.replace(old_title=old_title, new_class_dct=new_cls)

    def update(self, rhs):
        for new_cls_dct in rhs:
            self.add(new_cls_dct)
