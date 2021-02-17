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
from supervisely_lib.api.api import Api
from supervisely_lib.sly_logger import logger
from supervisely_lib.io.fs_cache import FileCache


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
    '''
    This is a class for creating and using Dataset objects. Here is where your labeled and unlabeled images and other
    files live. There is no more levels: images or videos are directly attached to a dataset. Dataset is a unit of work.
    All images or videos are directly attached to a dataset. A dataset is some sort of data folder with stuff to annotate.
    '''
    item_dir_name = 'img'
    annotation_class = Annotation

    def __init__(self, directory: str, mode: OpenMode):
        '''
        :param directory: path to the directory where the data set will be saved or where it will be loaded from
        :param mode: OpenMode class object which determines in what mode to work with the dataset
        '''
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
        '''
        :return: path to the directory with images in dataset
        '''
        # @TODO: deprecated method, should be private and be renamed in future
        return os.path.join(self.directory, self.item_dir_name)

    @property
    def ann_dir(self):
        '''
        :return: path to the directory with annotations in dataset
        '''
        return os.path.join(self.directory, 'ann')

    @staticmethod
    def _has_valid_ext(path: str) -> bool:
        '''
        The function _has_valid_ext checks if a given file has a supported extension('.jpg', '.jpeg', '.mpo', '.bmp', '.png', '.webp')
        :param path: the path to the file
        :return: bool (True if a given file has a supported extension, False - in otherwise)
        '''
        return sly_image.has_valid_ext(path)

    def _read(self):
        '''
        Fills out the dictionary items: item file name -> annotation file name. Checks item and annotation directoris existing and dataset not empty.
        Consistency checks. Every image must have an annotation, and the correspondence must be one to one.
        If not - it generate exception error.
        '''
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
        '''
        Creates a leaf directory and all intermediate ones for items and annatations.
        '''
        mkdir(self.ann_dir)
        mkdir(self.item_dir)

    def item_exists(self, item_name):
        '''
        Checks if given item name belongs to items of dataset
        :param item_name: str
        :return: bool
        '''
        return item_name in self._item_to_ann

    def get_item_path(self, item_name):
        '''
        :param item_name: str
        :return: str (path to given item), generate exception error if item not found in dataset
        '''
        return self.get_img_path(item_name)

    def get_img_path(self, item_name):
        '''
        :param item_name: str
        :return: str (path to given image), generate exception error if image not found in dataset
        '''
        # @TODO: deprecated method, should be private and be renamed in future
        if not self.item_exists(item_name):
            raise RuntimeError('Item {} not found in the project.'.format(item_name))
        return os.path.join(self.item_dir, item_name)

    def get_ann_path(self, item_name):
        '''
        :param item_name: str
        :return: str (path to given annotation), generate exception error if annotation not found in dataset
        '''
        ann_path = self._item_to_ann.get(item_name, None)
        if ann_path is None:
            raise RuntimeError('Item {} not found in the project.'.format(item_name))
        return os.path.join(self.ann_dir, ann_path)

    def add_item_file(self, item_name, item_path, ann=None, _validate_item=True, _use_hardlink=False):
        '''
        Add given item file to dataset items directory, and add given annatation to dataset annotations dir(if ann=None
        empty annotation will be create). Generate exception error if item_name already exists in dataset or item name has unsupported extension
        :param item_name: str
        :param item_path: str
        :param ann: Annotation class object, str, dict, None (generate exception error if param type is another)
        :param _validate_item: bool
        :param _use_hardlink: bool
        '''
        self._add_item_file(item_name, item_path, _validate_item=_validate_item, _use_hardlink=_use_hardlink)
        self._add_ann_by_type(item_name, ann)

    def add_item_np(self, item_name, img, ann=None):
        '''
        Write given image(RGB format(numpy matrix)) to dataset items directory, and add given annatation to dataset annotations dir(if ann=None
        empty annotation will be create). Generate exception error if item_name already exists in dataset or item name has unsupported extension
        :param item_name: str
        :param img: image in RGB format(numpy matrix)
        :param ann: Annotation class object, str, dict, None (generate exception error if param type is another)
        '''
        self._add_img_np(item_name, img)
        self._add_ann_by_type(item_name, ann)

    def add_item_raw_bytes(self, item_name, item_raw_bytes, ann=None):
        '''
        Write given binary object to dataset items directory, and add given annatation to dataset annotations dir(if ann=None
        empty annotation will be create). Generate exception error if item_name already exists in dataset or item name has unsupported extension.
        Make sure we actually received a valid image file, clean it up and fail if not so.
        :param item_name: str
        :param item_raw_bytes: binary object
        :param ann: Annotation class object, str, dict, None (generate exception error if param type is another)
        '''
        self._add_item_raw_bytes(item_name, item_raw_bytes)
        self._add_ann_by_type(item_name, ann)

    def _get_empty_annotaion(self, item_name):
        '''
        Create empty annotation from given item. Generate exception error if item not found in project
        :param item_name: str
        :return: Annotation class object
        '''
        img_size = sly_image.read(self.get_img_path(item_name)).shape[:2]
        return self.annotation_class(img_size)

    def _add_ann_by_type(self, item_name, ann):
        '''
        Add given annatation to dataset annotations dir and to dictionary items: item file name -> annotation file name
        :param item_name: str
        :param ann: Annotation class object, str, dict, None (generate exception error if param type is another)
        '''
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
        '''
        Generate exception error if item name already exists in dataset or has unsupported extension
        :param item_name: str
        '''
        if item_name in self._item_to_ann:
            raise RuntimeError('Item {!r} already exists in dataset {!r}.'.format(item_name, self.name))
        if not self._has_valid_ext(item_name):
            raise RuntimeError('Item name {!r} has unsupported extension.'.format(item_name))

    def _add_item_raw_bytes(self, item_name, item_raw_bytes):
        '''
        Write given binary object to dataset items directory, Generate exception error if item_name already exists in
        dataset or item name has unsupported extension. Make sure we actually received a valid image file, clean it up and fail if not so.
        :param item_name: str
        :param item_raw_bytes: binary object
        '''
        self._check_add_item_name(item_name)
        dst_img_path = os.path.join(self.item_dir, item_name)
        with open(dst_img_path, 'wb') as fout:
            fout.write(item_raw_bytes)
        self._validate_added_item_or_die(dst_img_path)

    def generate_item_path(self, item_name):
        '''
        :param item_name: str
        :return: str (full path to the given item)
        '''
        return os.path.join(self.item_dir, item_name)

    def _add_img_np(self, item_name, img):
        '''
        Write given image(RGB format(numpy matrix)) to dataset items directory. Generate exception error if item_name
        already exists in dataset or item name has unsupported extension
        :param item_name: str
        :param img: image in RGB format(numpy matrix)
        '''
        self._check_add_item_name(item_name)
        dst_img_path = os.path.join(self.item_dir, item_name)
        sly_image.write(dst_img_path, img)

    def _add_item_file(self, item_name, item_path, _validate_item=True, _use_hardlink=False):
        self._add_img_file(item_name, item_path, _validate_item, _use_hardlink)

    def _add_img_file(self, item_name, img_path, _validate_img=True, _use_hardlink=False):
        '''
        Add given item file to dataset items directory. Generate exception error if item_name already exists in dataset
        or item name has unsupported extension
        :param item_name: str
        :param img_path: str
        :param _validate_img: bool
        :param _use_hardlink: bool
        '''
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
        '''
        Make sure we actually received a valid image file, clean it up and fail if not so
        :param item_path: str
        '''
        # Make sure we actually received a valid image file, clean it up and fail if not so.
        try:
            sly_image.validate_format(item_path)
        except (sly_image.UnsupportedImageFormat, sly_image.ImageReadException):
            os.remove(item_path)
            raise

    def set_ann(self, item_name: str, ann):
        '''
        Save given annotation with given name to dataset annotations dir in json format.
        :param item_name: str
        :param ann: Annotation class object (Generate exception error if not so)
        '''
        if type(ann) is not self.annotation_class:
            raise TypeError("Type of 'ann' have to be Annotation, not a {}".format(type(ann)))
        dst_ann_path = self.get_ann_path(item_name)
        dump_json_file(ann.to_json(), dst_ann_path, indent=4)

    def set_ann_file(self, item_name: str, ann_path: str):
        '''
        Copy given annotation with given name to dataset annotations dir
        :param item_name: str
        :param ann_path: str (Generate exception error if not so)
        '''
        if type(ann_path) is not str:
            raise TypeError("Annotation path should be a string, not a {}".format(type(ann_path)))
        dst_ann_path = self.get_ann_path(item_name)
        copy_file(ann_path, dst_ann_path)

    def set_ann_dict(self, item_name: str, ann: dict):
        '''
        Save given annotation with given name to dataset annotations dir in json format.
        :param item_name: str
        :param ann: dict (json format)
        '''
        if type(ann) is not dict:
            raise TypeError("Ann should be a dict, not a {}".format(type(ann)))
        dst_ann_path = self.get_ann_path(item_name)
        dump_json_file(ann, dst_ann_path, indent=4)

    def get_item_paths(self, item_name) -> ItemPaths:
        '''
        Create ItemPaths object with passes to items and annotation dirs with given name
        :param item_name: str
        :return: ItemPaths class object
        '''
        return ItemPaths(img_path=self.get_img_path(item_name), ann_path=self.get_ann_path(item_name))

    def __len__(self):
        return len(self._item_to_ann)

    def __next__(self):
        for item_name in self._item_to_ann.keys():
            yield item_name

    def __iter__(self):
        return next(self)


class Project:
    '''
    This is a class for creating and using Project objects. You can think of a Project as a superfolder with data and
    meta information.
    '''
    dataset_class = Dataset
    class DatasetDict(KeyIndexedCollection):
        item_type = Dataset

    def __init__(self, directory, mode: OpenMode):
        '''
        :param directory: path to the directory where the project will be saved or where it will be loaded from
        :param mode: OpenMode class object which determines in what mode to work with the project (generate exception error if not so)
        '''
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
        '''
        :return: total number of items in project
        '''
        return sum(len(ds) for ds in self._datasets)

    def _get_project_meta_path(self):
        '''
        :return: str (path to project meta file(meta.json))
        '''
        return os.path.join(self.directory, 'meta.json')

    def _read(self):
        '''
        Download project from given project directory. Checks item and annotation directoris existing and dataset not empty.
        Consistency checks. Every image must have an annotation, and the correspondence must be one to one.
        '''
        meta_json = load_json_file(self._get_project_meta_path())
        self._meta = ProjectMeta.from_json(meta_json)

        possible_datasets = get_subdirs(self.directory)
        for ds_name in possible_datasets:
            current_dataset = self.dataset_class(os.path.join(self.directory, ds_name), OpenMode.READ)
            self._datasets = self._datasets.add(current_dataset)

        if self.total_items == 0:
            raise RuntimeError('Project is empty')

    def _create(self):
        '''
        Creates a leaf directory and empty meta.json file. Generate exception error if project directory already exists and is not empty.
        '''
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
        '''
        Save given meta to project dir in json format.
        :param new_meta: ProjectMeta class object
        '''
        self._meta = new_meta
        dump_json_file(self.meta.to_json(), self._get_project_meta_path(), indent=4)

    def __iter__(self):
        return next(self)

    def __next__(self):
        for dataset in self._datasets:
            yield dataset

    def create_dataset(self, ds_name):
        '''
        Creates a leaf directory with given name and all intermediate ones for items and annatations. Add new dataset
        to the collection of all datasets in project
        :param ds_name: str
        :return: Dataset class object
        '''
        ds = self.dataset_class(os.path.join(self.directory, ds_name), OpenMode.CREATE)
        self._datasets = self._datasets.add(ds)
        return ds

    def _add_item_file_to_dataset(self, ds, item_name, item_paths, _validate_item, _use_hardlink):
        '''
        Add item file and annotation from given name and path to given dataset items directory. Generate exception error if item_name already exists in dataset or item name has unsupported extension
        :param ds: Dataset class object
        :param item_name: str
        :param item_paths: ItemPaths object
        :param _validate_item: bool
        :param _use_hardlink: bool
        '''
        ds.add_item_file(item_name, item_paths.img_path,
                         ann=item_paths.ann_path, _validate_item=_validate_item, _use_hardlink=_use_hardlink)

    def copy_data(self, dst_directory, dst_name=None, _validate_item=True, _use_hardlink=False):
        '''
        Make copy of project in given directory.
        :param dst_directory: str
        :param dst_name: str
        :param _validate_item: bool
        :param _use_hardlink: bool
        :return: Project class object
        '''
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
        '''
        Split given path to project on parent directory and directory where project is located
        :param project_dir: str
        :return: str, str
        '''
        #alternative implementation
        #temp_parent_dir = os.path.dirname(parent_dir)
        #temp_name = os.path.basename(parent_dir)

        parent_dir, pr_name = os.path.split(project_dir.rstrip('/'))
        if not pr_name:
            raise RuntimeError('Unable to determine project name.')
        return parent_dir, pr_name


def read_single_project(dir, project_class=Project):
    '''
    Read project from given ditectory. Generate exception error if given dir contains more than one subdirectory
    :param dir: str
    :param project_class: Project class object type
    :return: Project class object
    '''
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


def _download_project(api, project_id, dest_dir, dataset_ids=None, log_progress=False, batch_size=10):
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

        for batch in batched(images, batch_size):
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


def upload_project(dir, api, workspace_id, project_name=None, log_progress=True):
    project_fs = read_single_project(dir)
    if project_name is None:
        project_name = project_fs.name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(workspace_id, project_name, )
    api.project.update_meta(project.id, project_fs.meta.to_json())

    for dataset_fs in project_fs.datasets:
        dataset = api.dataset.create(project.id, dataset_fs.name)

        names, img_paths, ann_paths = [], [], []
        for item_name in dataset_fs:
            img_path, ann_path = dataset_fs.get_item_paths(item_name)
            names.append(item_name)
            img_paths.append(img_path)
            ann_paths.append(ann_path)

        progress_cb = None
        if log_progress:
            ds_progress = Progress('Uploading images to dataset {!r}'.format(dataset.name), total_cnt=len(img_paths))
            progress_cb = ds_progress.iters_done_report
        img_infos = api.image.upload_paths(dataset.id, names, img_paths, progress_cb)
        image_ids = [img_info.id for img_info in img_infos]

        if log_progress:
            ds_progress = Progress('Uploading annotations to dataset {!r}'.format(dataset.name), total_cnt=len(img_paths))
            progress_cb = ds_progress.iters_done_report
        api.annotation.upload_paths(image_ids, ann_paths, progress_cb)

    return project.id, project.name


def download_project(api: Api, project_id, dest_dir, dataset_ids=None, log_progress=False, batch_size=10,
                     cache: FileCache = None, progress_cb=None):
    if cache is None:
        _download_project(api, project_id, dest_dir, dataset_ids, log_progress, batch_size)
    else:
        _download_project_optimized(api, project_id, dest_dir, dataset_ids, cache, progress_cb)


def _download_project_optimized(api: Api, project_id, project_dir, datasets_whitelist=None, cache=None, progress_cb=None):
    project_info = api.project.get_info_by_id(project_id)
    project_id = project_info.id
    logger.info(f"Annotations are not cached (always download latest version from server)")
    project_fs = Project(project_dir, OpenMode.CREATE)
    meta = ProjectMeta.from_json(api.project.get_meta(project_id))
    project_fs.set_meta(meta)
    for dataset_info in api.dataset.get_list(project_id):
        dataset_name = dataset_info.name
        dataset_id = dataset_info.id
        need_download = True
        if datasets_whitelist is not None and dataset_id not in datasets_whitelist:
            need_download = False
        if need_download is True:
            dataset = project_fs.create_dataset(dataset_name)
            _download_dataset(api, dataset, dataset_id, cache=cache, progress_cb=progress_cb)


def _split_images_by_cache(images, cache):
    images_to_download = []
    images_in_cache = []
    images_cache_paths = []
    for image in images:
        _, effective_ext = os.path.splitext(image.name)
        if len(effective_ext) == 0:
            # Fallback for the old format where we were cutting off extensions from image names.
            effective_ext = image.ext
        cache_path = cache.check_storage_object(image.hash, effective_ext)
        if cache_path is None:
            images_to_download.append(image)
        else:
            images_in_cache.append(image)
            images_cache_paths.append(cache_path)
    return images_to_download, images_in_cache, images_cache_paths


def _maybe_append_image_extension(name, ext):
    name_split = os.path.splitext(name)
    if name_split[1] == '':
        normalized_ext = ('.' + ext).replace('..', '.')
        result = name + normalized_ext
        sly_image.validate_ext(result)
    else:
        result = name
    return result


def _download_dataset(api: Api, dataset, dataset_id, cache=None, progress_cb=None):
    images = api.image.get_list(dataset_id)

    images_to_download = images

    # copy images from cache to task folder and download corresponding annotations
    if cache:
        images_to_download, images_in_cache, images_cache_paths = _split_images_by_cache(images, cache)
        if len(images_to_download) + len(images_in_cache) != len(images):
            raise RuntimeError("Error with images cache during download. Please contact support.")
        logger.info(f"Download dataset: {dataset.name}", extra={"total": len(images),
                                                                "in cache": len(images_in_cache),
                                                                "to download": len(images_to_download)})
        if len(images_in_cache) > 0:
            img_cache_ids = [img_info.id for img_info in images_in_cache]
            ann_info_list = api.annotation.download_batch(dataset_id, img_cache_ids, progress_cb)
            img_name_to_ann = {ann.image_id: ann.annotation for ann in ann_info_list}
            for batch in batched(list(zip(images_in_cache, images_cache_paths)), batch_size=50):
                for img_info, img_cache_path in batch:
                    item_name = _maybe_append_image_extension(img_info.name, img_info.ext)
                    dataset.add_item_file(item_name, img_cache_path, img_name_to_ann[img_info.id], _validate_item=False,
                                          _use_hardlink=True)
                progress_cb(len(batch))

    # download images from server
    if len(images_to_download) > 0:
        # prepare lists for api methods
        img_ids = []
        img_paths = []
        for img_info in images_to_download:
            img_ids.append(img_info.id)
            # TODO download to a temp file and use dataset api to add the image to the dataset.
            img_paths.append(
                os.path.join(dataset.img_dir, _maybe_append_image_extension(img_info.name, img_info.ext)))

        # download annotations
        ann_info_list = api.annotation.download_batch(dataset_id, img_ids, progress_cb)
        img_name_to_ann = {ann.image_id: ann.annotation for ann in ann_info_list}
        api.image.download_paths(dataset_id, img_ids, img_paths, progress_cb)
        for img_info, img_path in zip(images_to_download, img_paths):
            dataset.add_item_file(img_info.name, img_path, img_name_to_ann[img_info.id])

        if cache:
            img_hashes = [img_info.hash for img_info in images_to_download]
            cache.write_objects(img_paths, img_hashes)