# coding: utf-8

from collections import namedtuple
import os
import json
from enum import Enum

from supervisely_lib.annotation.annotation import Annotation, ANN_EXT
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection, KeyObject
from supervisely_lib.imaging import image
from supervisely_lib.io.fs import list_files, get_file_name, get_file_ext, mkdir, copy_file, get_subdirs, dir_exists
from supervisely_lib.io.json import dump_json_file, load_json_file

ItemPaths = namedtuple('ItemPaths', ['img_path', 'ann_path'])


class OpenMode(Enum):
    READ = 1
    CREATE = 2


class Dataset(KeyObject):
    def __init__(self, directory: str, mode: OpenMode):
        if type(mode) is not OpenMode:
            raise TypeError("Argument \'mode\' has type {!r}. Correct type is OpenMode".format(type(mode)))

        self._directory = directory
        self._items_exts = {}  # item_name -> image extension

        project_dir, ds_name = os.path.split(directory.rstrip('/'))
        self._project_dir = project_dir
        self._name = ds_name

        if mode is OpenMode.READ:
            self._read()
        else:
            self._create()

    @property
    def name(self):
        return self._name

    def key(self):
        return self.name

    @property
    def directory(self):
        return self._directory

    @property
    def img_dir(self):
        return os.path.join(self.directory, 'img')

    @property
    def ann_dir(self):
        return os.path.join(self.directory, 'ann')

    def _read(self):
        if not dir_exists(self.img_dir):
            raise FileNotFoundError('Image directory not found: {!r}'.format(self.img_dir))
        if not dir_exists(self.ann_dir):
            raise FileNotFoundError('Annotation directory not found: {!r}'.format(self.ann_dir))

        ann_paths = list_files(self.ann_dir, [ANN_EXT])
        img_paths = list_files(self.img_dir, image.SUPPORTED_IMG_EXTS)

        ann_names = set(get_file_name(path) for path in ann_paths)
        img_names = {get_file_name(path): get_file_ext(path) for path in img_paths}

        if len(img_names) == 0 or len(ann_names) == 0:
            raise RuntimeError('Dataset {!r} is empty'.format(self.name))
        if ann_names != set(img_names.keys()):
            raise RuntimeError('File names in dataset {!r} are inconsistent'.format(self.name))

        self._items_exts = img_names

    def _create(self):
        mkdir(self.ann_dir)
        mkdir(self.img_dir)

    def _item_ext_or_die(self, item_name):
        item_ext = self._items_exts.get(item_name)
        if item_ext is None:
            raise RuntimeError('Item {} not found in the project.'.format(item_name))
        return item_ext

    def item_exists(self, item_name):
        return self._items_exts.get(item_name) is not None

    def get_img_path(self, item_name):
        item_ext = self._item_ext_or_die(item_name)
        return os.path.join(self.img_dir, item_name + item_ext)

    # TODO clean up public usages of this
    def deprecated_make_img_path(self, item_name, img_ext):
        img_ext = ('.' + img_ext).replace('..', '.')
        image.validate_ext(img_ext)
        return os.path.join(self.img_dir, item_name + img_ext)

    def get_ann_path(self, item_name):
        _ = self._item_ext_or_die(item_name)  # Check that the item actually exists in the dataset.
        return os.path.join(self.ann_dir, item_name + ANN_EXT)

    def add_item_file(self, item_name, img_path, ann=None):
        self._add_img_file(item_name, img_path)
        self._set_ann_by_type(item_name, ann)

    def add_item_np(self, item_name, img, img_ext, ann=None):
        self._add_img_np(item_name, img, img_ext)
        self._set_ann_by_type(item_name, ann)

    def _set_ann_by_type(self, item_name, ann):
        if ann is None:
            img_path = self.deprecated_make_img_path(item_name, self._items_exts[item_name])
            img_size = image.read(img_path).shape[:2]
            self.set_ann(item_name, Annotation(img_size))
        elif type(ann) is Annotation:
            self.set_ann(item_name, ann)
        elif type(ann) is str:
            self.set_ann_file(item_name, ann)
        elif type(ann) is dict:
            self.set_ann_dict(item_name, ann)
        else:
            raise TypeError("Unsupported type {!r} for ann argument".format(type(ann)))

    def _add_img_np(self, item_name, img, img_ext):
        dst_img_path = self.deprecated_make_img_path(item_name, img_ext)
        image.write(dst_img_path, img)
        self._items_exts[item_name] = img_ext

    def _add_img_file(self, item_name, img_path):
        img_ext = get_file_ext(img_path)
        dst_img_path = self.deprecated_make_img_path(item_name, img_ext)
        if img_path != dst_img_path:  # used only for agent + api during download project
            copy_file(img_path, dst_img_path)
        self._items_exts[item_name] = img_ext

    def set_ann(self, item_name: str, ann: Annotation):
        if type(ann) is not Annotation:
            raise TypeError("Type of 'ann' have to be Annotation, not a {}".format(type(ann)))
        dst_ann_path = self.get_ann_path(item_name)
        dump_json_file(ann.to_json(), dst_ann_path)

    def set_ann_file(self, item_name: str, ann_path: str):
        if type(ann_path) is not str:
            raise TypeError("Annotation path should be a string, not a {}".format(type(ann_path)))
        dst_ann_path = self.get_ann_path(item_name)
        copy_file(ann_path, dst_ann_path)

    def set_ann_dict(self, item_name: str, ann: dict):
        if type(ann) is not dict:
            raise TypeError("Ann should be a dict, not a {}".format(type(ann)))
        dst_ann_path = self.get_ann_path(item_name)
        dump_json_file(ann, dst_ann_path)

    def get_item_paths(self, item_name) -> ItemPaths:
        return ItemPaths(img_path=self.get_img_path(item_name), ann_path=self.get_ann_path(item_name))

    def __len__(self):
        return len(self._items_exts)

    def __next__(self):
        for item_name in self._items_exts.keys():
            yield item_name

    def __iter__(self):
        return next(self)


class Project:
    class DatasetDict(KeyIndexedCollection):
        item_type = Dataset

    def __init__(self, directory, mode: OpenMode):
        if type(mode) is not OpenMode:
            raise TypeError("Argument \'mode\' has type {!r}. Correct type is OpenMode".format(type(mode)))

        parent_dir, name = Project._parse_path(directory)
        self._parent_dir = parent_dir
        self._name = name
        self._datasets = Project.DatasetDict()  # ds_name -> dataset object
        self._meta = None

        if mode is OpenMode.READ:
            self._read()
        else:
            self._create()

    @property
    def parent_dir(self):
        return self._parent_dir

    @property
    def name(self):
        return self._name

    @property
    def datasets(self):
        return self._datasets

    @property
    def meta(self):
        return self._meta

    @property
    def directory(self):
        return os.path.join(self.parent_dir, self.name)

    @property
    def total_items(self):
        return sum(len(ds) for ds in self._datasets)

    def _get_project_meta_path(self):
        return os.path.join(self.directory, 'meta.json')

    def _read(self):
        meta_json = load_json_file(self._get_project_meta_path())
        self._meta = ProjectMeta.from_json(meta_json)

        possible_datasets = get_subdirs(self.directory)
        for ds_name in possible_datasets:
            current_dataset = Dataset(os.path.join(self.directory, ds_name), OpenMode.READ)
            self._datasets = self._datasets.add(current_dataset)

        if self.total_items == 0:
            raise RuntimeError('Project is empty')

    def _create(self):
        if dir_exists(self.directory):
            raise RuntimeError("Can not create new project {!r}. Directory {!r} already exists".format(self.name, self.directory))
        mkdir(self.directory)
        self.set_meta(ProjectMeta())

    def validate(self):
        # @TODO: validation here
        pass

    def set_meta(self, new_meta):
        self._meta = new_meta
        json.dump(self.meta.to_json(), open(self._get_project_meta_path(), 'w'))

    def __iter__(self):
        return next(self)

    def __next__(self):
        for dataset in self._datasets:
            yield dataset

    def create_dataset(self, ds_name):
        ds = Dataset(os.path.join(self.directory, ds_name), OpenMode.CREATE)
        self._datasets = self._datasets.add(ds)
        return ds

    def copy_data(self, dst_directory, dst_name=None):
        dst_name = dst_name if dst_name is not None else self.name
        new_project = Project(os.path.join(dst_directory, dst_name), OpenMode.CREATE)
        new_project.set_meta(self.meta)

        for ds in self:
            new_ds = new_project.create_dataset(ds.name)
            for item_name in ds:
                item_paths = ds.get_item_paths(item_name)
                new_ds.add_item_file(item_name, item_paths.img_path, ann=item_paths.ann_path)
        return new_project

    @staticmethod
    def _parse_path(project_dir):
        #alternative implementation
        #temp_parent_dir = os.path.dirname(parent_dir)
        #temp_name = os.path.basename(parent_dir)

        parent_dir, pr_name = os.path.split(project_dir.rstrip('/'))
        if not pr_name:
            raise RuntimeError('Unable to determine project name.')
        return parent_dir, pr_name


def read_single_project(dir):
    projects_in_dir = get_subdirs(dir)
    if len(projects_in_dir) != 1:
        raise RuntimeError('Found {} dirs instead of 1'.format(len(projects_in_dir)))
    return Project(os.path.join(dir, projects_in_dir[0]), OpenMode.READ)

