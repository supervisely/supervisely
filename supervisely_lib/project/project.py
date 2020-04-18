# coding: utf-8

from collections import namedtuple
import os
import json
from enum import Enum

from supervisely_lib.annotation.annotation import Annotation, ANN_EXT
from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection, KeyObject
from supervisely_lib.imaging import image as sly_image
from supervisely_lib.io.fs import list_files, list_files_recursively, mkdir, copy_file, get_subdirs, dir_exists
from supervisely_lib.io.json import dump_json_file, load_json_file
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.task.progress import Progress
from supervisely_lib._utils import batched
from supervisely_lib.io.fs import file_exists

# @TODO: rename img_path to item_path
ItemPaths = namedtuple('ItemPaths', ['img_path', 'ann_path'])


class OpenMode(Enum):
    READ = 1
    CREATE = 2


def _get_effective_ann_name(img_name, ann_names):
    new_format_name = img_name + ANN_EXT
    if new_format_name in ann_names:
        return new_format_name
    else:
        old_format_name = os.path.splitext(img_name)[0] + ANN_EXT
        return old_format_name if (old_format_name in ann_names) else None


class Dataset(KeyObject):
    item_dir_name = 'img'
    annotation_class = Annotation

    def __init__(self, directory: str, mode: OpenMode):
        if type(mode) is not OpenMode:
            raise TypeError("Argument \'mode\' has type {!r}. Correct type is OpenMode".format(type(mode)))

        self._directory = directory
        self._item_to_ann = {} # item file name -> annotation file name

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
    def item_dir(self):
        return self.img_dir

    @property
    def img_dir(self):
        # @TODO: deprecated method, should be private and be renamed in future
        return os.path.join(self.directory, self.item_dir_name)

    @property
    def ann_dir(self):
        return os.path.join(self.directory, 'ann')

    @staticmethod
    def _has_valid_ext(path: str) -> bool:
        return sly_image.has_valid_ext(path)

    def _read(self):
        if not dir_exists(self.item_dir):
            raise FileNotFoundError('Item directory not found: {!r}'.format(self.item_dir))
        if not dir_exists(self.ann_dir):
            raise FileNotFoundError('Annotation directory not found: {!r}'.format(self.ann_dir))

        raw_ann_paths = list_files(self.ann_dir, [ANN_EXT])
        img_paths = list_files(self.item_dir, filter_fn=self._has_valid_ext)

        raw_ann_names = set(os.path.basename(path) for path in raw_ann_paths)
        img_names = [os.path.basename(path) for path in img_paths]

        if len(img_names) == 0 or len(raw_ann_names) == 0:
            raise RuntimeError('Dataset {!r} is empty'.format(self.name))

        # Consistency checks. Every image must have an annotation, and the correspondence must be one to one.
        effective_ann_names = set()
        for img_name in img_names:
            ann_name = _get_effective_ann_name(img_name, raw_ann_names)
            if ann_name is None:
                raise RuntimeError('Item {!r} in dataset {!r} does not have a corresponding annotation file.'.format(
                    img_name, self.name))
            if ann_name in effective_ann_names:
                raise RuntimeError('Annotation file {!r} in dataset {!r} matches two different image files.'.format(
                    ann_name, self.name))
            effective_ann_names.add(ann_name)
            self._item_to_ann[img_name] = ann_name

    def _create(self):
        mkdir(self.ann_dir)
        mkdir(self.item_dir)

    def item_exists(self, item_name):
        return item_name in self._item_to_ann

    def get_item_path(self, item_name):
        return self.get_img_path(item_name)

    def get_img_path(self, item_name):
        # @TODO: deprecated method, should be private and be renamed in future
        if not self.item_exists(item_name):
            raise RuntimeError('Item {} not found in the project.'.format(item_name))
        return os.path.join(self.item_dir, item_name)

    def get_ann_path(self, item_name):
        ann_path = self._item_to_ann.get(item_name, None)
        if ann_path is None:
            raise RuntimeError('Item {} not found in the project.'.format(item_name))
        return os.path.join(self.ann_dir, ann_path)

    def add_item_file(self, item_name, item_path, ann=None, _validate_item=True, _use_hardlink=False):
        self._add_item_file(item_name, item_path, _validate_item=_validate_item, _use_hardlink=_use_hardlink)
        self._add_ann_by_type(item_name, ann)

    def add_item_np(self, item_name, img, ann=None):
        self._add_img_np(item_name, img)
        self._add_ann_by_type(item_name, ann)

    def add_item_raw_bytes(self, item_name, item_raw_bytes, ann=None):
        self._add_item_raw_bytes(item_name, item_raw_bytes)
        self._add_ann_by_type(item_name, ann)

    def _get_empty_annotaion(self, item_name):
        img_size = sly_image.read(self.get_img_path(item_name)).shape[:2]
        return self.annotation_class(img_size)

    def _add_ann_by_type(self, item_name, ann):
        # This is a new-style annotation name, so if there was no image with this name yet, there should not have been
        # an annotation either.
        self._item_to_ann[item_name] = item_name + ANN_EXT
        if ann is None:
            self.set_ann(item_name, self._get_empty_annotaion(item_name))
        elif type(ann) is self.annotation_class:
            self.set_ann(item_name, ann)
        elif type(ann) is str:
            self.set_ann_file(item_name, ann)
        elif type(ann) is dict:
            self.set_ann_dict(item_name, ann)
        else:
            raise TypeError("Unsupported type {!r} for ann argument".format(type(ann)))

    def _check_add_item_name(self, item_name):
        if item_name in self._item_to_ann:
            raise RuntimeError('Item {!r} already exists in dataset {!r}.'.format(item_name, self.name))
        if not self._has_valid_ext(item_name):
            raise RuntimeError('Item name {!r} has unsupported extension.'.format(item_name))

    def _add_item_raw_bytes(self, item_name, item_raw_bytes):
        self._check_add_item_name(item_name)
        dst_img_path = os.path.join(self.item_dir, item_name)
        with open(dst_img_path, 'wb') as fout:
            fout.write(item_raw_bytes)
        self._validate_added_item_or_die(dst_img_path)

    def generate_item_path(self, item_name):
        return os.path.join(self.item_dir, item_name)

    def _add_img_np(self, item_name, img):
        self._check_add_item_name(item_name)
        dst_img_path = os.path.join(self.item_dir, item_name)
        sly_image.write(dst_img_path, img)

    def _add_item_file(self, item_name, item_path, _validate_item=True, _use_hardlink=False):
        self._add_img_file(item_name, item_path, _validate_item, _use_hardlink)

    def _add_img_file(self, item_name, img_path, _validate_img=True, _use_hardlink=False):
        # @TODO: deprecated method, should be private and be (refactored, renamed) in future
        self._check_add_item_name(item_name)
        dst_img_path = os.path.join(self.item_dir, item_name)
        if img_path != dst_img_path and img_path is not None:  # used only for agent + api during download project + None to optimize internal usage
            hardlink_done = False
            if _use_hardlink:
                try:
                    os.link(img_path, dst_img_path)
                    hardlink_done = True
                except OSError:
                    pass
            if not hardlink_done:
                copy_file(img_path, dst_img_path)
            if _validate_img:
                self._validate_added_item_or_die(img_path)

    @staticmethod
    def _validate_added_item_or_die(item_path):
        # Make sure we actually received a valid image file, clean it up and fail if not so.
        try:
            sly_image.validate_format(item_path)
        except (sly_image.UnsupportedImageFormat, sly_image.ImageReadException):
            os.remove(item_path)
            raise

    def set_ann(self, item_name: str, ann):
        if type(ann) is not self.annotation_class:
            raise TypeError("Type of 'ann' have to be Annotation, not a {}".format(type(ann)))
        dst_ann_path = self.get_ann_path(item_name)
        dump_json_file(ann.to_json(), dst_ann_path, indent=4)

    def set_ann_file(self, item_name: str, ann_path: str):
        if type(ann_path) is not str:
            raise TypeError("Annotation path should be a string, not a {}".format(type(ann_path)))
        dst_ann_path = self.get_ann_path(item_name)
        copy_file(ann_path, dst_ann_path)

    def set_ann_dict(self, item_name: str, ann: dict):
        if type(ann) is not dict:
            raise TypeError("Ann should be a dict, not a {}".format(type(ann)))
        dst_ann_path = self.get_ann_path(item_name)
        dump_json_file(ann, dst_ann_path, indent=4)

    def get_item_paths(self, item_name) -> ItemPaths:
        return ItemPaths(img_path=self.get_img_path(item_name), ann_path=self.get_ann_path(item_name))

    def __len__(self):
        return len(self._item_to_ann)

    def __next__(self):
        for item_name in self._item_to_ann.keys():
            yield item_name

    def __iter__(self):
        return next(self)


class Project:
    dataset_class = Dataset
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
            current_dataset = self.dataset_class(os.path.join(self.directory, ds_name), OpenMode.READ)
            self._datasets = self._datasets.add(current_dataset)

        if self.total_items == 0:
            raise RuntimeError('Project is empty')

    def _create(self):
        if dir_exists(self.directory):
            if len(list_files_recursively(self.directory)) > 0:
                raise RuntimeError(
                    "Cannot create new project {!r}. Directory {!r} already exists and is not empty".format(
                        self.name, self.directory))
        else:
            mkdir(self.directory)
        self.set_meta(ProjectMeta())

    def validate(self):
        # @TODO: validation here
        pass

    def set_meta(self, new_meta):
        self._meta = new_meta
        dump_json_file(self.meta.to_json(), self._get_project_meta_path(), indent=4)

    def __iter__(self):
        return next(self)

    def __next__(self):
        for dataset in self._datasets:
            yield dataset

    def create_dataset(self, ds_name):
        ds = self.dataset_class(os.path.join(self.directory, ds_name), OpenMode.CREATE)
        self._datasets = self._datasets.add(ds)
        return ds

    def _add_item_file_to_dataset(self, ds, item_name, item_paths, _validate_item, _use_hardlink):
        ds.add_item_file(item_name, item_paths.img_path,
                         ann=item_paths.ann_path, _validate_item=_validate_item, _use_hardlink=_use_hardlink)

    def copy_data(self, dst_directory, dst_name=None, _validate_item=True, _use_hardlink=False):
        dst_name = dst_name if dst_name is not None else self.name
        new_project = Project(os.path.join(dst_directory, dst_name), OpenMode.CREATE)
        new_project.set_meta(self.meta)

        for ds in self:
            new_ds = new_project.create_dataset(ds.name)
            for item_name in ds:
                item_paths = ds.get_item_paths(item_name)
                self._add_item_file_to_dataset(new_ds, item_name, item_paths, _validate_item, _use_hardlink)
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


def read_single_project(dir, project_class=Project):
    projects_in_dir = get_subdirs(dir)
    if len(projects_in_dir) != 1:
        raise RuntimeError('Found {} dirs instead of 1'.format(len(projects_in_dir)))

    project_dir = os.path.join(dir, projects_in_dir[0])
    try:
        project_fs = project_class(project_dir, OpenMode.READ)
    except Exception as e:
        projects_in_dir = get_subdirs(project_dir)
        if len(projects_in_dir) != 1:
            raise e
        project_dir = os.path.join(project_dir, projects_in_dir[0])
        project_fs = project_class(project_dir, OpenMode.READ)

    return project_fs


def download_project(api, project_id, dest_dir, dataset_ids=None, log_progress=False):
    dataset_ids = set(dataset_ids) if (dataset_ids is not None) else None
    project_fs = Project(dest_dir, OpenMode.CREATE)
    meta = ProjectMeta.from_json(api.project.get_meta(project_id))
    project_fs.set_meta(meta)

    for dataset_info in api.dataset.get_list(project_id):
        dataset_id = dataset_info.id
        if dataset_ids is not None and dataset_id not in dataset_ids:
            continue

        dataset_fs = project_fs.create_dataset(dataset_info.name)
        images = api.image.get_list(dataset_id)

        ds_progress = None
        if log_progress:
            ds_progress = Progress(
                'Downloading dataset: {!r}'.format(dataset_info.name), total_cnt=len(images))

        for batch in batched(images):
            image_ids = [image_info.id for image_info in batch]
            image_names = [image_info.name for image_info in batch]

            # download images in numpy format
            batch_imgs_bytes = api.image.download_bytes(dataset_id, image_ids)

            # download annotations in json format
            ann_infos = api.annotation.download_batch(dataset_id, image_ids)
            ann_jsons = [ann_info.annotation for ann_info in ann_infos]

            for name, img_bytes, ann in zip(image_names, batch_imgs_bytes, ann_jsons):
                dataset_fs.add_item_raw_bytes(name, img_bytes, ann)

            if log_progress:
                ds_progress.iters_done_report(len(batch))
