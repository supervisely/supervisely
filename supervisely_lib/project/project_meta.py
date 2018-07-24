# coding: utf-8

import os.path as osp
from enum import Enum

from ..figure.fig_classes import FigClasses
from ..utils.os_utils import ensure_base_path
from ..utils.json_utils import *


class ProjectMetaFmt(Enum):
    V1_CLASSES = 1  # old format, list of classes
    V2_META = 2  # curr format, classes, plain tags (for imgs & objs)


_DEFAULT_OUT_FMT = ProjectMetaFmt.V2_META


class ProjectMeta(object):
    fmt_to_fname = {
        ProjectMetaFmt.V1_CLASSES: 'classes.json',
        ProjectMetaFmt.V2_META: 'meta.json',
    }

    # py_container is native Python container like appropriate dict or list
    def __init__(self, py_container=None):
        self.classes = FigClasses()
        self.img_tags = set()
        self.obj_tags = set()

        if type(py_container) is list:
            self._in_fmt = ProjectMetaFmt.V1_CLASSES
            self.classes = FigClasses(classes_lst=py_container)

        elif type(py_container) is dict:
            self._in_fmt = ProjectMetaFmt.V2_META
            self.classes = FigClasses(classes_lst=py_container['classes'])
            self.img_tags = set(py_container['tags_images'])
            self.obj_tags = set(py_container['tags_objects'])

        elif py_container is None:
            self._in_fmt = None  # empty meta

        else:
            raise RuntimeError('Wrong meta object type.')

    @property
    def input_format(self):
        return self._in_fmt

    def update(self, rhs):
        self.classes.update(rhs.classes)
        self.img_tags.update(rhs.img_tags)
        self.obj_tags.update(rhs.obj_tags)

    def to_py_container(self, out_fmt=_DEFAULT_OUT_FMT):
        if out_fmt == ProjectMetaFmt.V1_CLASSES:
            res = self.classes.py_container
        elif out_fmt == ProjectMetaFmt.V2_META:
            res = {
                'classes': self.classes.py_container,
                'tags_images': list(self.img_tags),
                'tags_objects': list(self.obj_tags),
            }
        else:
            raise NotImplementedError()
        return res

    def to_json_file(self, fpath, out_fmt=_DEFAULT_OUT_FMT):
        ensure_base_path(fpath)
        json_dump(self.to_py_container(out_fmt), fpath)

    def to_json_str(self, out_fmt=_DEFAULT_OUT_FMT):
        res = json_dumps(self.to_py_container(out_fmt))
        return res

    def to_dir(self, dir_path, out_fmt=_DEFAULT_OUT_FMT):
        fpath = self.dir_path_to_fpath(dir_path, out_fmt)
        self.to_json_file(fpath, out_fmt)

    @classmethod
    def dir_path_to_fpath(cls, dir_path, fmt=_DEFAULT_OUT_FMT):
        fname = cls.fmt_to_fname[fmt]
        return osp.join(dir_path, fname)

    @classmethod
    def from_json_file(cls, fpath):
        py_container = json_load(fpath)
        return cls(py_container)

    @classmethod
    def from_json_str(cls, s):
        py_container = json_loads(s)
        return cls(py_container)

    @classmethod
    def find_in_dir(cls, dir_path):
        for fmt in ProjectMetaFmt:
            fpath = cls.dir_path_to_fpath(dir_path, fmt=fmt)
            if osp.isfile(fpath):
                return fpath
        return None

    @classmethod
    def from_dir(cls, dir_path):
        fpath = cls.find_in_dir(dir_path)
        if not fpath:
            raise RuntimeError('File with meta not found in dir: {}'.format(dir_path))
        return cls.from_json_file(fpath)
