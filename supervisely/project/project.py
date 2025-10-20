# coding: utf-8

from __future__ import annotations

import asyncio
import gc
import io
import json
import os
import pickle
import random
import shutil
from collections import defaultdict, namedtuple
from enum import Enum
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import aiofiles
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

import supervisely as sly
from supervisely._utils import (
    abs_url,
    batched,
    get_or_create_event_loop,
    is_development,
    removesuffix,
    snake_to_human,
)
from supervisely.annotation.annotation import ANN_EXT, Annotation, TagCollection
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.api.api import Api, ApiContext, ApiField
from supervisely.api.image_api import (
    OFFSETS_PKL_BATCH_SIZE,
    OFFSETS_PKL_SUFFIX,
    BlobImageInfo,
    ImageInfo,
)
from supervisely.api.project_api import ProjectInfo
from supervisely.collection.key_indexed_collection import (
    KeyIndexedCollection,
    KeyObject,
)
from supervisely.geometry.bitmap import Bitmap
from supervisely.imaging import image as sly_image
from supervisely.io.fs import (
    clean_dir,
    copy_file,
    copy_file_async,
    dir_empty,
    dir_exists,
    ensure_base_path,
    file_exists,
    get_file_name_with_ext,
    list_dir_recursively,
    list_files,
    list_files_recursively,
    mkdir,
    silent_remove,
    subdirs_tree,
)
from supervisely.io.fs_cache import FileCache
from supervisely.io.json import dump_json_file, dump_json_file_async, load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly

TF_BLOB_DIR = "blob-files"  # directory for project blob files in team files


class CustomUnpickler(pickle.Unpickler):
    """
    Custom Unpickler for loading pickled objects of the same class with differing definitions.
    Handles cases where a class object is reconstructed using a newer definition with additional fields
    or an outdated definition missing some fields.
    Supports loading namedtuple objects with missing or extra fields.
    """

    def __init__(self, file, **kwargs):
        super().__init__(file, **kwargs)
        self.warned_classes = set()  # To prevent multiple warnings for the same class
        self.sdk_update_notified = False

    def find_class(self, module, name):
        prefix = "Pickled"
        cls = super().find_class(module, name)
        if hasattr(cls, "_fields") and "Info" in cls.__name__:
            orig_new = cls.__new__

            def new(cls, *args, **kwargs):
                orig_class_name = cls.__name__[len(prefix) :]
                # Case when new definition of class has more fields than the old one
                if len(args) < len(cls._fields):
                    default_values = cls._field_defaults
                    # Set missed attrs to None
                    num_missing = len(cls._fields) - len(args)
                    args = list(args) + [None] * num_missing
                    # Replace only the added None values with default values where applicable
                    args[-num_missing:] = [
                        (
                            default_values.get(field, arg)
                            if arg is None and field in default_values
                            else arg
                        )
                        for field, arg in zip(cls._fields[-num_missing:], args[-num_missing:])
                    ]
                    if orig_class_name not in self.warned_classes:
                        new_fields = cls._fields[len(cls._fields) - num_missing :]
                        logger.warning(
                            f"New fields {new_fields} for the '{orig_class_name}' class objects are set to their default values or None due to an updated definition of this class."
                        )
                        self.warned_classes.add(orig_class_name)
                # Case when the object of new class definition creating within old class definition
                elif len(args) > len(cls._fields):
                    end_index = len(args)
                    args = args[: len(cls._fields)]
                    if orig_class_name not in self.warned_classes:
                        logger.warning(
                            f"Extra fields idx {list(range(len(cls._fields), end_index))} are ignored for '{orig_class_name}' class objects due to an outdated class definition"
                        )
                        self.warned_classes.add(orig_class_name)
                        if not self.sdk_update_notified:
                            logger.warning(
                                "It is recommended to update the SDK version to restore the project version correctly."
                            )
                            self.sdk_update_notified = True
                return orig_new(cls, *args, **kwargs)

            # Create a new subclass dynamically to prevent redefining the current class
            NewCls = type(f"{prefix}{cls.__name__}", (cls,), {"__new__": new})
            return NewCls

        return cls


# @TODO: rename img_path to item_path (maybe convert namedtuple to class and create fields and props)
class ItemPaths(NamedTuple):
    #: :class:`str`: Full image file path of item
    img_path: str

    #: :class:`str`: Full annotation file path of item
    ann_path: str


class ItemInfo(NamedTuple):
    #: :class:`str`: Item's dataset name
    dataset_name: str

    #: :class:`str`: Item name
    name: str

    #: :class:`str`: Full image file path of item
    img_path: str

    #: :class:`str`: Full annotation file path of item
    ann_path: str


class OpenMode(Enum):
    """
    Defines the mode of using the :class:`Project<Project>` and :class:`Dataset<Dataset>`.
    """

    #: :class:`int`: READ open mode.
    #: Loads project from given project directory. Checks that item and annotation directories
    #: exist and dataset is not empty. Consistency checks. Checks that every image has
    #: an annotation and the correspondence is one to one.
    READ = 1

    #: :class:`int`: CREATE open mode.
    #: Creates a leaf directory and empty meta.json file. Generates error if
    #: project directory already exists and is not empty.
    CREATE = 2


def _get_effective_ann_name(img_name, ann_names):
    new_format_name = img_name + ANN_EXT
    if new_format_name in ann_names:
        return new_format_name
    else:
        old_format_name = os.path.splitext(img_name)[0] + ANN_EXT
        return old_format_name if (old_format_name in ann_names) else None


class Dataset(KeyObject):
    """
    Dataset is where your labeled and unlabeled images and other data files live. :class:`Dataset<Dataset>` object is immutable.

    :param directory: Path to dataset directory.
    :type directory: str
    :param mode: Determines working mode for the given dataset.
    :type mode: :class:`OpenMode<OpenMode>`, optional. If not provided, dataset_id must be provided.
    :param parents: List of parent directories, e.g. ["ds1", "ds2", "ds3"].
    :type parents: List[str]
    :param dataset_id: Dataset ID if the Dataset is opened in API mode.
        If dataset_id is specified then api must be specified as well.
    :type dataset_id: Optional[int]
    :param api: API object if the Dataset is opened in API mode.
    :type api: Optional[:class:`Api<supervis
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"

        # To open dataset locally in read mode
        ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

        # To open dataset on API
        api = sly.Api.from_env()
        ds = sly.Dataset(dataset_path, dataset_id=1, api=api)
    """

    annotation_class = Annotation
    item_info_class = ImageInfo

    item_dir_name = "img"
    ann_dir_name = "ann"
    item_info_dir_name = "img_info"
    seg_dir_name = "seg"
    meta_dir_name = "meta"
    datasets_dir_name = "datasets"
    blob_dir_name = "blob"

    def __init__(
        self,
        directory: str,
        mode: Optional[OpenMode] = None,
        parents: Optional[List[str]] = None,
        dataset_id: Optional[int] = None,
        api: Optional[sly.Api] = None,
    ):
        if dataset_id is not None:
            raise NotImplementedError(
                "Opening dataset from the API is not implemented yet. Please use the local mode "
                "by providing the 'directory' and 'mode' arguments."
                "This feature will be available later."
            )
        if type(mode) is not OpenMode and mode is not None:
            raise TypeError(
                "Argument 'mode' has type {!r}. Correct type is OpenMode".format(type(mode))
            )
        if mode is None and dataset_id is None:
            raise ValueError("Either 'mode' or 'dataset_id' must be provided")
        if dataset_id is not None and api is None:
            raise ValueError("Argument 'api' must be provided if 'dataset_id' is provided")

        self.parents = parents or []

        self.dataset_id = dataset_id
        self._api = api

        self._directory = directory
        self._item_to_ann = {}  # item file name -> annotation file name

        parts = directory.split(os.path.sep)
        if self.datasets_dir_name not in parts:
            project_dir, ds_name = os.path.split(directory.rstrip("/"))
            full_ds_name = short_ds_name = ds_name
        else:
            nested_ds_dir_index = parts.index(self.datasets_dir_name)
            ds_dir_index = nested_ds_dir_index - 1

            project_dir = os.path.join(*parts[:ds_dir_index])
            full_ds_name = os.path.join(
                *[p for p in parts[ds_dir_index:] if p != self.datasets_dir_name]
            )
            short_ds_name = os.path.basename(directory)

        self._project_dir = project_dir
        self._name = full_ds_name
        self._short_name = short_ds_name
        self._blob_offset_paths = []

        if self.dataset_id is not None:
            self._read_api()
        elif mode is OpenMode.READ:
            self._read()
        else:
            self._create()

    @classmethod
    def ignorable_dirs(cls) -> List[str]:
        ignorable_dirs = [getattr(cls, attr) for attr in dir(cls) if attr.endswith("_dir_name")]
        return [p for p in ignorable_dirs if isinstance(p, str)]

    @classmethod
    def datasets_dir(cls) -> List[str]:
        return cls.datasets_dir_name

    @property
    def project_dir(self) -> str:
        """
        Path to the project containing the dataset.

        :return: Path to the project.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds0"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)
            print(ds.project_dir)
            # Output: "/home/admin/work/supervisely/projects/lemons_annotated"
        """
        return self._project_dir

    @property
    def name(self) -> str:
        """
        Full Dataset name, which includes it's parents,
        e.g. ds1/ds2/ds3.

        Use :attr:`short_name` to get only the name of the dataset.

        :return: Dataset Name.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)
            print(ds.name)
            # Output: "ds1"
        """
        return self._name

    @property
    def short_name(self) -> str:
        """
        Short dataset name, which does not include it's parents.
        To get the full name of the dataset, use :attr:`name`.

        :return: Dataset Name.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)
            print(ds.name)
            # Output: "ds1"
        """
        return self._short_name

    @property
    def path(self) -> str:
        """Returns a relative local path to the dataset.

        :return: Relative local path to the dataset.
        :rtype: :class:`str`
        """
        return self._get_dataset_path(self.short_name, self.parents)

    @staticmethod
    def _get_dataset_path(dataset_name: str, parents: List[dir]):
        """Returns a relative local path to the dataset.

        :param dataset_name: Dataset name.
        :type dataset_name: :class:`str`
        """
        relative_path = os.path.sep.join(f"{parent}/datasets" for parent in parents)
        return os.path.join(relative_path, dataset_name)

    def key(self):
        # TODO: add docstring
        return self.name

    @property
    def directory(self) -> str:
        """
        Path to the dataset directory.

        :return: Path to the dataset directory.
        :rtype: :class:`str`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.directory)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1'
        """
        return self._directory

    @property
    def item_dir(self) -> str:
        """
        Path to the dataset items directory.

        :return: Path to the dataset directory with items.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.item_dir)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/img'
        """
        return os.path.join(self.directory, self.item_dir_name)

    @property
    def img_dir(self) -> str:
        """
        Path to the dataset images directory.
        Property is alias of item_dir.

        :return: Path to the dataset directory with images.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.img_dir)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/img'
        """
        return self.item_dir

    @property
    def ann_dir(self) -> str:
        """
        Path to the dataset annotations directory.

        :return: Path to the dataset directory with annotations.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.ann_dir)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/ann'
        """
        return os.path.join(self.directory, self.ann_dir_name)

    @property
    def img_info_dir(self):
        """
        Path to the dataset image info directory.
        Property is alias of item_info_dir.

        :return: Path to the dataset directory with images info.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.img_info_dir)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/img_info'
        """
        return self.item_info_dir

    @property
    def item_info_dir(self):
        """
        Path to the dataset item info directory.

        :return: Path to the dataset directory with items info.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.item_info_dir)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/img_info'
        """
        return os.path.join(self.directory, self.item_info_dir_name)

    @property
    def seg_dir(self):
        """
        Path to the dataset segmentation masks directory.

        :return: Path to the dataset directory with masks.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.seg_dir)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/seg'
        """
        return os.path.join(self.directory, self.seg_dir_name)

    @property
    def meta_dir(self):
        """
        Path to the dataset segmentation masks directory.

        :return: Path to the dataset directory with masks.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.meta_dir)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/meta'
        """
        return os.path.join(self.directory, self.meta_dir_name)

    @property
    def blob_offsets(self):
        """
        List of paths to the dataset blob offset files.

        :return: List of paths to the dataset blob offset files.
        :rtype: :class:`List[str]`
        """
        return self._blob_offset_paths

    @blob_offsets.setter
    def blob_offsets(self, value: List[str]):
        """
        Set the list of paths to the dataset blob offset files.
        """
        self._blob_offset_paths = value

    @classmethod
    def _has_valid_ext(cls, path: str) -> bool:
        """
        The function _has_valid_ext checks if a given file has a supported extension('.jpg', '.jpeg', '.mpo', '.bmp', '.png', '.webp')
        :param path: the path to the file
        :return: bool (True if a given file has a supported extension, False - in otherwise)
        """
        return sly_image.has_valid_ext(path)

    def _read(self):
        """
        Fills out the dictionary items: item file name -> annotation file name. Checks item and annotation directoris existing and dataset not empty.
        Consistency checks. Every item must have an annotation, and the correspondence must be one to one.
        If not - it generate exception error.
        """
        blob_offset_paths = list_files(
            self.directory, filter_fn=lambda x: x.endswith(OFFSETS_PKL_SUFFIX)
        )
        has_blob_offsets = len(blob_offset_paths) > 0

        if not dir_exists(self.item_dir) and not has_blob_offsets:
            raise FileNotFoundError("Item directory not found: {!r}".format(self.item_dir))
        if not dir_exists(self.ann_dir):
            raise FileNotFoundError("Annotation directory not found: {!r}".format(self.ann_dir))

        raw_ann_paths = list_files(self.ann_dir, [ANN_EXT])
        raw_ann_names = set(os.path.basename(path) for path in raw_ann_paths)

        if dir_exists(self.item_dir):
            img_paths = list_files(self.item_dir, filter_fn=self._has_valid_ext)
            img_names = [os.path.basename(path) for path in img_paths]
        else:
            img_names = []

        # If we have blob offset files, add the image names from those
        if has_blob_offsets:
            self.blob_offsets = blob_offset_paths
            for offset_file_path in self.blob_offsets:
                try:
                    blob_img_info_lists = BlobImageInfo.load_from_pickle_generator(offset_file_path)
                    for blob_img_info_list in blob_img_info_lists:
                        for blob_img_info in blob_img_info_list:
                            img_names.append(blob_img_info.name)
                except Exception as e:
                    logger.warning(f"Failed to read blob offset file {offset_file_path}: {str(e)}")

        if len(img_names) == 0 and len(raw_ann_names) == 0:
            logger.debug(f"Dataset '{self.name}' is empty")
            # raise RuntimeError("Dataset {!r} is empty".format(self.name))

        if len(img_names) == 0:  # items_names polyfield
            img_names = [os.path.splitext(ann_name)[0] for ann_name in raw_ann_names]

        # Consistency checks. Every image must have an annotation, and the correspondence must be one to one.
        effective_ann_names = set()
        for img_name in img_names:
            ann_name = _get_effective_ann_name(img_name, raw_ann_names)
            if ann_name is None:
                raise RuntimeError(
                    "Item {!r} in dataset {!r} does not have a corresponding annotation file.".format(
                        img_name, self.name
                    )
                )
            if ann_name in effective_ann_names:
                raise RuntimeError(
                    "Annotation file {!r} in dataset {!r} matches two different image files.".format(
                        ann_name, self.name
                    )
                )
            effective_ann_names.add(ann_name)
            self._item_to_ann[img_name] = ann_name

    def _read_api(self) -> None:
        """Method to read the dataset, which opened from the API."""
        self._image_infos = self._api.image.get_list(self.dataset_id)
        img_names = [img_info.name for img_info in self._image_infos]
        for img_name in img_names:
            ann_name = f"{img_name}.json"
            self._item_to_ann[img_name] = ann_name

    @property
    def image_infos(self) -> List[ImageInfo]:
        """If the dataset is opened from the API, returns the list of ImageInfo objects.
        Otherwise raises an exception.

        :raises: ValueError: If the dataset is opened in local mode.
        :return: List of ImageInfo objects.
        :rtype: List[:class:`ImageInfo`]
        """
        if not self.dataset_id:
            raise ValueError(
                "This dataset was open in local mode. It does not have access to the API."
            )
        return self._image_infos

    def _create(self):
        """
        Creates a leaf directory and all intermediate ones for items and annotations.
        """
        mkdir(self.ann_dir)
        mkdir(self.item_dir)

    def get_items_names(self) -> list:
        """
        List of dataset item names.

        :return: List of item names.
        :rtype: :class:`list` [ :class:`str` ]
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_item_names())
            # Output: ['IMG_0002.jpg', 'IMG_0005.jpg', 'IMG_0008.jpg', ...]
        """
        return list(self._item_to_ann.keys())

    def item_exists(self, item_name: str) -> bool:
        """
        Checks if given item name belongs to the dataset.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: True if item exist, otherwise False.
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            ds.item_exists("IMG_0748")      # False
            ds.item_exists("IMG_0748.jpeg") # True
        """
        return item_name in self._item_to_ann

    def get_item_path(self, item_name: str) -> str:
        """
        Path to the given item.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: Path to the given item.
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_item_path("IMG_0748"))
            # Output: RuntimeError: Item IMG_0748 not found in the project.

            print(ds.get_item_path("IMG_0748.jpeg"))
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/img/IMG_0748.jpeg'
        """
        if not self.item_exists(item_name):
            raise RuntimeError("Item {} not found in the project.".format(item_name))

        return os.path.join(self.item_dir, item_name)

    def get_img_path(self, item_name: str) -> str:
        """
        Path to the given image.
        Method is alias of get_item_path(item_name).

        :param item_name: Image name.
        :type item_name: :class:`str`
        :return: Path to the given image.
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_img_path("IMG_0748"))
            # Output: RuntimeError: Item IMG_0748 not found in the project.

            print(ds.get_img_path("IMG_0748.jpeg"))
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/ann/IMG_0748.jpeg.json'
        """
        return self.get_item_path(item_name)

    def get_ann(self, item_name, project_meta: ProjectMeta) -> Annotation:
        """
        Read annotation of item from json.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param project_meta: ProjectMeta object.
        :type project_meta: :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`
        :return: Annotation object.
        :rtype: :class:`Annotation<supervisely.annotation.annotation.Annotation>`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            project = sly.Project(project_path, sly.OpenMode.READ)

            ds = project.datasets.get('ds1')

            annotation = ds.get_ann("IMG_0748", project.meta)
            # Output: RuntimeError: Item IMG_0748 not found in the project.

            annotation = ds.get_ann("IMG_0748.jpeg", project.meta)
            print(annotation.to_json())
            # Output: {
            #     "description": "",
            #     "size": {
            #         "height": 500,
            #         "width": 700
            #     },
            #     "tags": [],
            #     "objects": [],
            #     "customBigData": {}
            # }
        """
        ann_path = self.get_ann_path(item_name)
        return self.annotation_class.load_json_file(ann_path, project_meta)

    def get_ann_path(self, item_name: str) -> str:
        """
        Path to the given annotation json file.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: Path to the given annotation json file.
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_ann_path("IMG_0748"))
            # Output: RuntimeError: Item IMG_0748 not found in the project.

            print(ds.get_ann_path("IMG_0748.jpeg"))
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/ann/IMG_0748.jpeg.json'
        """
        ann_path = self._item_to_ann.get(item_name, None)
        if ann_path is None:
            raise RuntimeError("Item {} not found in the project.".format(item_name))

        ann_path = ann_path.strip("/")
        return os.path.join(self.ann_dir, ann_path)

    def get_img_info_path(self, img_name: str) -> str:
        """
        Get path to the image info json file without checking if the file exists.
        Method is alias of get_item_info_path(item_name).

        :param item_name: Image name.
        :type item_name: :class:`str`
        :return: Path to the given image info json file.
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if image not found in the project.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_img_info_path("IMG_0748"))
            # Output: RuntimeError: Item IMG_0748 not found in the project.

            print(ds.get_img_info_path("IMG_0748.jpeg"))
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/img_info/IMG_0748.jpeg.json'
        """
        return self.get_item_info_path(img_name)

    def get_item_info_path(self, item_name: str) -> str:
        """
        Get path to the item info json file without checking if the file exists.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: Path to the given item info json file.
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_item_info_path("IMG_0748"))
            # Output: RuntimeError: Item IMG_0748 not found in the project.

            print(ds.get_item_info_path("IMG_0748.jpeg"))
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/img_info/IMG_0748.jpeg.json'
        """
        info_path = self._item_to_ann.get(item_name, None)
        if info_path is None:
            raise RuntimeError("Item {} not found in the project.".format(item_name))

        return os.path.join(self.item_info_dir, info_path)

    def get_item_meta_path(self, item_name: str) -> str:
        """
        Get path to the item info json file without checking if the file exists.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: Path to the given item info json file.
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_item_info_path("IMG_0748"))
            # Output: RuntimeError: Item IMG_0748 not found in the project.

            print(ds.get_item_info_path("IMG_0748.jpeg"))
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/img_info/IMG_0748.jpeg.json'
        """
        meta_path = self._item_to_ann.get(item_name, None)

        return os.path.join(self.meta_dir, meta_path)

    def get_image_info(self, item_name: str) -> ImageInfo:
        """
        Information for Item with given name.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: ImageInfo object.
        :rtype: :class:`ImageInfo<supervisely.api.image_api.ImageInfo>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds0"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_image_info("IMG_0748.jpeg"))
            # Output:
            # ImageInfo(
            #     id=770915,
            #     name='IMG_0748.jpeg',
            #     link=None,
            #     hash='ZdpMD+ZMJx0R8BgsCzJcqM7qP4M8f1AEtoYc87xZmyQ=',
            #     mime='image/jpeg',
            #     ext='jpeg',
            #     size=148388,
            #     width=1067,
            #     height=800,
            #     labels_count=4,
            #     dataset_id=2532,
            #     created_at='2021-03-02T10:04:33.973Z',
            #     updated_at='2021-03-02T10:04:33.973Z',
            #     meta={},
            #     path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpeg',
            #     full_storage_url='http://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpeg'),
            #     tags=[]
            # )
        """
        return self.get_item_info(item_name)

    def get_item_info(self, item_name: str) -> ImageInfo:
        """
        Information for Item with given name.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: ImageInfo object.
        :rtype: :class:`ImageInfo<supervisely.api.image_api.ImageInfo>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds0"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_item_info("IMG_0748.jpeg"))
            # Output:
            # ImageInfo(
            #     id=770915,
            #     name='IMG_0748.jpeg',
            #     link=None,
            #     hash='ZdpMD+ZMJx0R8BgsCzJcqM7qP4M8f1AEtoYc87xZmyQ=',
            #     mime='image/jpeg',
            #     ext='jpeg',
            #     size=148388,
            #     width=1067,
            #     height=800,
            #     labels_count=4,
            #     dataset_id=2532,
            #     created_at='2021-03-02T10:04:33.973Z',
            #     updated_at='2021-03-02T10:04:33.973Z',
            #     meta={},
            #     path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpeg',
            #     full_storage_url='http://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpeg'),
            #     tags=[]
            # )
        """
        item_info_path = self.get_item_info_path(item_name)
        item_info_dict = load_json_file(item_info_path)
        item_info_named_tuple = namedtuple(self.item_info_class.__name__, item_info_dict)
        return item_info_named_tuple(**item_info_dict)

    def get_seg_path(self, item_name: str) -> str:
        """
        Get path to the png segmentation mask file without checking if the file exists.
        Use :class:`Project.to_segmentation_task()<supervisely.project.project.Project.to_segmentation_task>`
        to create segmentation masks from annotations in your project.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: Path to the given png mask file.
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_seg_path("IMG_0748"))
            # Output: RuntimeError: Item IMG_0748 not found in the project.

            print(ds.get_seg_path("IMG_0748.jpeg"))
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/seg/IMG_0748.jpeg.png'
        """
        ann_path = self._item_to_ann.get(item_name, None)
        if ann_path is None:
            raise RuntimeError("Item {} not found in the project.".format(item_name))
        return os.path.join(self.seg_dir, f"{item_name}.png")

    def add_item_file(
        self,
        item_name: str,
        item_path: str,
        ann: Optional[Union[Annotation, str]] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
        item_info: Optional[Union[ImageInfo, Dict, str]] = None,
        img_info: Optional[Union[ImageInfo, Dict, str]] = None,
    ) -> None:
        """
        Adds given item file to dataset items directory, and adds given annotation to dataset
        annotations directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param item_path: Path to the item.
        :type item_path: :class:`str`
        :param ann: Annotation object or path to annotation json file.
        :type ann: :class:`Annotation<supervisely.annotation.annotation.Annotation>` or :class:`str`, optional
        :param _validate_item: Checks input files format.
        :type _validate_item: :class:`bool`, optional
        :param _use_hardlink: If True creates a hardlink pointing to src named dst, otherwise don't.
        :type _use_hardlink: :class:`bool`, optional
        :param item_info: ImageInfo object or ImageInfo object converted to dict or path to item info json file for copying to dataset item info directory.
        :type item_info: :class:`ImageInfo<supervisely.api.image_api.ImageInfo>` or :class:`dict` or :class:`str`, optional
        :param img_info: Deprecated version of item_info parameter. Can be removed in future versions.
        :type img_info: :class:`ImageInfo<supervisely.api.image_api.ImageInfo>` or :class:`dict` or :class:`str`, optional
        :return: None
        :rtype: NoneType
        :raises: :class:`RuntimeError` if item_name already exists in dataset or item name has unsupported extension.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            ann = "/home/admin/work/supervisely/projects/lemons_annotated/ds1/ann/IMG_8888.jpeg.json"
            ds.add_item_file("IMG_8888.jpeg", "/home/admin/work/supervisely/projects/lemons_annotated/ds1/img/IMG_8888.jpeg", ann=ann)
            print(ds.item_exists("IMG_8888.jpeg"))
            # Output: True
        """
        # item_path is None when image is cached
        if item_path is None and ann is None and img_info is None:
            raise RuntimeError("No item_path or ann or img_info provided.")

        if item_info is not None and img_info is not None:
            raise RuntimeError(
                "At least one parameter of two (item_info and img_info) must be None."
            )

        if img_info is not None:
            logger.warn(
                "img_info parameter of add_item_file() method is deprecated and can be removed in future versions. Use item_info parameter instead."
            )
            item_info = img_info

        self._add_item_file(
            item_name,
            item_path,
            _validate_item=_validate_item,
            _use_hardlink=_use_hardlink,
        )
        self._add_ann_by_type(item_name, ann)
        self._add_item_info(item_name, item_info)

    def add_item_np(
        self,
        item_name: str,
        img: np.ndarray,
        ann: Optional[Union[Annotation, str]] = None,
        img_info: Optional[Union[ImageInfo, Dict, str]] = None,
    ) -> None:
        """
        Adds given numpy matrix as an image to dataset items directory, and adds given annotation to dataset ann directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param img: numpy Image matrix in RGB format.
        :type img: np.ndarray
        :param ann: Annotation object or path to annotation json file.
        :type ann: :class:`Annotation<supervisely.annotation.annotation.Annotation>` or :class:`str`, optional
        :param img_info: ImageInfo object or ImageInfo object converted to dict or path to item info json file for copying to dataset item info directory.
        :type img_info: :class:`ImageInfo<supervisely.api.image_api.ImageInfo>` or :class:`dict` or :class:`str`, optional
        :return: None
        :rtype: NoneType
        :raises: :class:`RuntimeError` if item_name already exists in dataset or item name has unsupported extension
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            img_path = "/home/admin/Pictures/Clouds.jpeg"
            img_np = sly.image.read(img_path)
            ds.add_item_np("IMG_050.jpeg", img_np)
            print(ds.item_exists("IMG_050.jpeg"))
            # Output: True
        """
        if img is None and ann is None and img_info is None:
            raise RuntimeError("No img or ann or img_info provided.")

        self._add_img_np(item_name, img)
        self._add_ann_by_type(item_name, ann)
        self._add_item_info(item_name, img_info)

    def add_item_raw_bytes(
        self,
        item_name: str,
        item_raw_bytes: bytes,
        ann: Optional[Union[Annotation, str]] = None,
        img_info: Optional[Union[ImageInfo, Dict, str]] = None,
    ) -> None:
        """
        Adds given binary object as an image to dataset items directory, and adds given annotation to dataset ann directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param item_raw_bytes: Binary object.
        :type item_raw_bytes: :class:`bytes`
        :param ann: Annotation object or path to annotation json file.
        :type ann: :class:`Annotation<supervisely.annotation.annotation.Annotation>` or :class:`str`, optional
        :param img_info: ImageInfo object or ImageInfo object converted to dict or path to item info json file for copying to dataset item info directory.
        :type img_info: :class:`ImageInfo<supervisely.api.image_api.ImageInfo>` or :class:`dict` or :class:`str`, optional
        :return: None
        :rtype: NoneType
        :raises: :class:`RuntimeError` if item_name already exists in dataset or item name has unsupported extension
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            img_path = "/home/admin/Pictures/Clouds.jpeg"
            img_np = sly.image.read(img_path)
            img_bytes = sly.image.write_bytes(img_np, "jpeg")
            ds.add_item_raw_bytes("IMG_050.jpeg", img_bytes)
            print(ds.item_exists("IMG_050.jpeg"))
            # Output: True
        """
        if item_raw_bytes is None and ann is None and img_info is None:
            raise RuntimeError("No item_raw_bytes or ann or img_info provided.")

        self._add_item_raw_bytes(item_name, item_raw_bytes)
        self._add_ann_by_type(item_name, ann)
        self._add_item_info(item_name, img_info)

    def get_classes_stats(
        self,
        project_meta: Optional[ProjectMeta] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        if project_meta is None:
            project = Project(self.project_dir, OpenMode.READ)
            project_meta = project.meta
        class_items = {}
        class_objects = {}
        class_figures = {}
        for obj_class in project_meta.obj_classes:
            class_items[obj_class.name] = 0
            class_objects[obj_class.name] = 0
            class_figures[obj_class.name] = 0
        for item_name in self:
            item_ann = self.get_ann(item_name, project_meta)
            item_class = {}
            for label in item_ann.labels:
                class_objects[label.obj_class.name] += 1
                item_class[label.obj_class.name] = True
            for obj_class in project_meta.obj_classes:
                if obj_class.name in item_class.keys():
                    class_items[obj_class.name] += 1

        result = {}
        if return_items_count:
            result["items_count"] = class_items
        if return_objects_count:
            result["objects_count"] = class_objects
        if return_figures_count:
            class_figures = class_objects.copy()  # for Images project
            result["figures_count"] = class_figures
        return result

    def _get_empty_annotaion(self, item_name):
        """
        Create empty annotation from given item. Generate exception error if item not found in project
        :param item_name: str
        :return: Annotation class object
        """
        img_size = sly_image.read(self.get_img_path(item_name)).shape[:2]
        return self.annotation_class(img_size)

    def _add_ann_by_type(self, item_name, ann):
        """
        Add given annotation to dataset annotations dir and to dictionary items: item file name -> annotation file name
        :param item_name: str
        :param ann: Annotation class object, str, dict, None (generate exception error if param type is another)
        """
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

    def _add_item_info(self, item_name, item_info=None):
        if item_info is None:
            return

        dst_info_path = self.get_item_info_path(item_name)
        ensure_base_path(dst_info_path)
        if type(item_info) is dict:
            dump_json_file(item_info, dst_info_path, indent=4)
        elif type(item_info) is str and os.path.isfile(item_info):
            shutil.copy(item_info, dst_info_path)
        else:
            # item info named tuple (ImageInfo, VideoInfo, PointcloudInfo, ..)
            dump_json_file(item_info._asdict(), dst_info_path, indent=4)

    async def _add_item_info_async(self, item_name, item_info=None):
        if item_info is None:
            return

        dst_info_path = self.get_item_info_path(item_name)
        ensure_base_path(dst_info_path)
        if type(item_info) is dict:
            dump_json_file(item_info, dst_info_path, indent=4)
        elif type(item_info) is str and os.path.isfile(item_info):
            shutil.copy(item_info, dst_info_path)
        else:
            # item info named tuple (ImageInfo, VideoInfo, PointcloudInfo, ..)
            dump_json_file(item_info._asdict(), dst_info_path, indent=4)

    def _check_add_item_name(self, item_name):
        """
        Generate exception error if item name already exists in dataset or has unsupported extension
        :param item_name: str
        """
        if item_name in self._item_to_ann:
            raise RuntimeError(
                "Item {!r} already exists in dataset {!r}.".format(item_name, self.name)
            )
        if not self._has_valid_ext(item_name):
            raise RuntimeError("Item name {!r} has unsupported extension.".format(item_name))

    def _add_item_raw_bytes(self, item_name, item_raw_bytes):
        """
        Write given binary object to dataset items directory, Generate exception error if item_name already exists in
        dataset or item name has unsupported extension. Make sure we actually received a valid image file, clean it up and fail if not so.
        :param item_name: str
        :param item_raw_bytes: binary object
        """
        if item_raw_bytes is None:
            return

        self._check_add_item_name(item_name)
        item_name = item_name.strip("/")
        dst_img_path = os.path.join(self.item_dir, item_name)
        os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
        with open(dst_img_path, "wb") as fout:
            fout.write(item_raw_bytes)
        self._validate_added_item_or_die(dst_img_path)

    async def _add_item_raw_bytes_async(self, item_name, item_raw_bytes):
        """
        Write given binary object to dataset items directory, Generate exception error if item_name already exists in
        dataset or item name has unsupported extension. Make sure we actually received a valid image file, clean it up and fail if not so.
        :param item_name: str
        :param item_raw_bytes: binary object
        """
        if item_raw_bytes is None:
            return

        self._check_add_item_name(item_name)
        item_name = item_name.strip("/")
        dst_img_path = os.path.join(self.item_dir, item_name)
        os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
        async with aiofiles.open(dst_img_path, "wb") as fout:
            await fout.write(item_raw_bytes)

        self._validate_added_item_or_die(dst_img_path)

    async def add_item_raw_bytes_async(
        self,
        item_name: str,
        item_raw_bytes: bytes,
        ann: Optional[Union[Annotation, str]] = None,
        img_info: Optional[Union[ImageInfo, Dict, str]] = None,
    ) -> None:
        """
        Adds given binary object as an image to dataset items directory, and adds given annotation to dataset ann directory.
        If ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param item_raw_bytes: Binary object.
        :type item_raw_bytes: :class:`bytes`
        :param ann: Annotation object or path to annotation json file.
        :type ann: :class:`Annotation<supervisely.annotation.annotation.Annotation>` or :class:`str`, optional
        :param img_info: ImageInfo object or ImageInfo object converted to dict or path to item info json file for copying to dataset item info directory.
        :type img_info: :class:`ImageInfo<supervisely.api.image_api.ImageInfo>` or :class:`dict` or :class:`str`, optional
        :return: None
        :rtype: NoneType
        :raises: :class:`RuntimeError` if item_name already exists in dataset or item name has unsupported extension
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely._utils import run_coroutine

            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            img_path = "/home/admin/Pictures/Clouds.jpeg"
            img_np = sly.image.read(img_path)
            img_bytes = sly.image.write_bytes(img_np, "jpeg")
            coroutine = ds.add_item_raw_bytes_async("IMG_050.jpeg", img_bytes)
            run_coroutine(coroutine)

            print(ds.item_exists("IMG_050.jpeg"))
            # Output: True
        """
        if item_raw_bytes is None and ann is None and img_info is None:
            raise RuntimeError("No item_raw_bytes or ann or img_info provided.")

        await self._add_item_raw_bytes_async(item_name, item_raw_bytes)
        await self._add_ann_by_type_async(item_name, ann)
        self._add_item_info(item_name, img_info)

    def generate_item_path(self, item_name: str) -> str:
        """
        Generates full path to the given item.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: Full path to the given item
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.generate_item_path("IMG_0748.jpeg"))
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/img/IMG_0748.jpeg'
        """
        # TODO: what the difference between this and ds.get_item_path() ?
        return os.path.join(self.item_dir, item_name)

    def _add_img_np(self, item_name, img):
        """
        Write given image(RGB format(numpy matrix)) to dataset items directory. Generate exception error if item_name
        already exists in dataset or item name has unsupported extension
        :param item_name: str
        :param img: image in RGB format(numpy matrix)
        """
        if img is None:
            return

        self._check_add_item_name(item_name)
        dst_img_path = os.path.join(self.item_dir, item_name)
        sly_image.write(dst_img_path, img)

    def _add_item_file(self, item_name, item_path, _validate_item=True, _use_hardlink=False):
        """
        Add given item file to dataset items directory. Generate exception error if item_name already exists in dataset
        or item name has unsupported extension
        :param item_name: str
        :param item_path: str
        :param _validate_item: bool
        :param _use_hardlink: bool
        """
        if item_path is None:
            return

        self._check_add_item_name(item_name)
        dst_item_path = os.path.join(self.item_dir, item_name)
        if (
            item_path != dst_item_path and item_path is not None
        ):  # used only for agent + api during download project + None to optimize internal usage
            hardlink_done = False
            if _use_hardlink:
                try:
                    os.link(item_path, dst_item_path)
                    hardlink_done = True
                except OSError:
                    pass
            if not hardlink_done:
                copy_file(item_path, dst_item_path)
            if _validate_item:
                self._validate_added_item_or_die(item_path)

    def _validate_added_item_or_die(self, item_path):
        """
        Make sure we actually received a valid image file, clean it up and fail if not so
        :param item_path: str
        """
        # Make sure we actually received a valid image file, clean it up and fail if not so.
        try:
            sly_image.validate_format(item_path)
        except (sly_image.UnsupportedImageFormat, sly_image.ImageReadException):
            os.remove(item_path)
            raise

    async def _validate_added_item_or_die_async(self, item_path):
        """
        Make sure we actually received a valid image file, clean it up and fail if not so
        :param item_path: str
        """
        # Make sure we actually received a valid image file, clean it up and fail if not so.
        try:
            sly_image.validate_format(item_path)
        except (sly_image.UnsupportedImageFormat, sly_image.ImageReadException):
            os.remove(item_path)
            raise

    def set_ann(self, item_name: str, ann: Annotation) -> None:
        """
        Replaces given annotation for given item name to dataset annotations directory in json format.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param ann: Annotation object.
        :type ann: :class:`Annotation<supervisely.annotation.annotation.Annotation>`
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            height, width = 500, 700
            new_ann = sly.Annotation((height, width))
            ds.set_ann("IMG_0748.jpeg", new_ann)
        """
        if type(ann) is not self.annotation_class:
            raise TypeError(
                f"Type of 'ann' should be {self.annotation_class.__name__}, not a {type(ann).__name__}"
            )
        dst_ann_path = self.get_ann_path(item_name)
        dump_json_file(ann.to_json(), dst_ann_path, indent=4)

    def set_ann_file(self, item_name: str, ann_path: str) -> None:
        """
        Replaces given annotation json file for given item name to dataset annotations directory in json format.

        :param item_name: Item Name.
        :type item_name: :class:`str`
        :param ann_path: Path to the :class:`Annotation<supervisely.annotation.annotation.Annotation>` json file.
        :type ann_path: :class:`str`
        :return: None
        :rtype: NoneType
        :raises: :class:`RuntimeError` if ann_path is not str
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            new_ann = "/home/admin/work/supervisely/projects/kiwi_annotated/ds1/ann/IMG_1812.jpeg.json"
            ds.set_ann_file("IMG_1812.jpeg", new_ann)
        """
        if type(ann_path) is not str:
            raise TypeError("Annotation path should be a string, not a {}".format(type(ann_path)))
        dst_ann_path = self.get_ann_path(item_name)
        copy_file(ann_path, dst_ann_path)

    def set_ann_dict(self, item_name: str, ann: Dict) -> None:
        """
        Replaces given annotation json for given item name to dataset annotations directory in json format.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param ann: :class:`Annotation<supervisely.annotation.annotation.Annotation>` as a dict in json format.
        :type ann: :class:`dict`
        :return: None
        :rtype: NoneType
        :raises: :class:`RuntimeError` if ann_path is not str
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            new_ann_json = {
                "description":"",
                "size":{
                    "height":500,
                    "width":700
                },
                "tags":[],
                "objects":[],
                "customBigData":{}
            }

            ds.set_ann_dict("IMG_8888.jpeg", new_ann_json)
        """
        if type(ann) is not dict:
            raise TypeError("Ann should be a dict, not a {}".format(type(ann)))
        dst_ann_path = self.get_ann_path(item_name)
        os.makedirs(os.path.dirname(dst_ann_path), exist_ok=True)
        dump_json_file(ann, dst_ann_path, indent=4)

    def get_item_paths(self, item_name: str) -> ItemPaths:
        """
        Generates :class:`ItemPaths<ItemPaths>` object with paths to item and annotation directories for item with given name.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: ItemPaths object
        :rtype: :class:`ItemPaths<ItemPaths>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            img_path, ann_path = dataset.get_item_paths("IMG_0748.jpeg")
            print("img_path:", img_path)
            print("ann_path:", ann_path)
            # Output:
            # img_path: /home/admin/work/supervisely/projects/lemons_annotated/ds1/img/IMG_0748.jpeg
            # ann_path: /home/admin/work/supervisely/projects/lemons_annotated/ds1/ann/IMG_0748.jpeg.json
        """
        return ItemPaths(
            img_path=self.get_item_path(item_name),
            ann_path=self.get_ann_path(item_name),
        )

    def __len__(self):
        return len(self._item_to_ann)

    def __next__(self):
        for item_name in self._item_to_ann.keys():
            yield item_name

    def __iter__(self):
        return next(self)

    def items(self) -> Generator[Tuple[str, str, str]]:
        """
        This method is used to iterate over dataset items, receiving item name, path to image and path to annotation
        json file. It is useful when you need to iterate over dataset items and get paths to images and annotations.

        :return: Generator object, that yields tuple of item name, path to image and path to annotation json file.
        :rtype: Generator[Tuple[str]]

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            input = "path/to/local/directory"
            # Creating Supervisely project from local directory.
            project = sly.Project(input, sly.OpenMode.READ)

            for dataset in project.datasets:
                for item_name, image_path, ann_path in dataset.items():
                    print(f"Item '{item_name}': image='{image_path}', ann='{ann_path}'")
        """
        for item_name in self._item_to_ann.keys():
            img_path, ann_path = self.get_item_paths(item_name)
            yield item_name, img_path, ann_path

    def delete_item(self, item_name: str) -> bool:
        """
        Delete image, image info and annotation from :class:`Dataset<Dataset>`.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: True if item was successfully deleted, False if item wasn't found in dataset.
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            print(dataset.delete_item("IMG_0748"))
            # Output: False

            print(dataset.delete_item("IMG_0748.jpeg"))
            # Output: True
        """
        if self.item_exists(item_name):
            data_path, ann_path = self.get_item_paths(item_name)
            item_info_path = self.get_item_info_path(item_name)
            silent_remove(data_path)
            silent_remove(ann_path)
            silent_remove(item_info_path)
            self._item_to_ann.pop(item_name)
            return True
        return False

    @staticmethod
    def get_url(project_id: int, dataset_id: int) -> str:
        """
        Get URL to dataset items list in Supervisely.

        :param project_id: :class:`Project<Project>` ID in Supervisely.
        :type project_id: :class:`int`
        :param dataset_id: :class:`Dataset<Dataset>` ID in Supervisely.
        :type dataset_id: :class:`int`
        :return: URL to dataset items list.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            from supervisely import Dataset

            project_id = 10093
            dataset_id = 45330
            ds_items_link = Dataset.get_url(project_id, dataset_id)

            print(ds_items_link)
            # Output: "/projects/10093/datasets/45330"
        """
        res = f"/projects/{project_id}/datasets/{dataset_id}"
        if is_development():
            res = abs_url(res)
        return res

    async def set_ann_file_async(self, item_name: str, ann_path: str) -> None:
        """
        Replaces given annotation json file for given item name to dataset annotations directory in json format.

        :param item_name: Item Name.
        :type item_name: :class:`str`
        :param ann_path: Path to the :class:`Annotation<supervisely.annotation.annotation.Annotation>` json file.
        :type ann_path: :class:`str`
        :return: None
        :rtype: NoneType
        :raises: :class:`RuntimeError` if ann_path is not str
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely._utils import run_coroutine

            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)
            new_ann = "/home/admin/work/supervisely/projects/kiwi_annotated/ds1/ann/IMG_1812.jpeg.json"

            coroutine = ds.set_ann_file_async("IMG_1812.jpeg", new_ann)
            run_coroutine(coroutine)
        """
        if type(ann_path) is not str:
            raise TypeError("Annotation path should be a string, not a {}".format(type(ann_path)))
        dst_ann_path = self.get_ann_path(item_name)
        await copy_file_async(ann_path, dst_ann_path)

    async def set_ann_dict_async(self, item_name: str, ann: Dict) -> None:
        """
        Replaces given annotation json for given item name to dataset annotations directory in json format.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param ann: :class:`Annotation<supervisely.annotation.annotation.Annotation>` as a dict in json format.
        :type ann: :class:`dict`
        :return: None
        :rtype: NoneType
        :raises: :class:`RuntimeError` if ann_path is not str
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely._utils import run_coroutine

            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            new_ann_json = {
                "description":"",
                "size":{
                    "height":500,
                    "width":700
                },
                "tags":[],
                "objects":[],
                "customBigData":{}
            }

            coroutine = ds.set_ann_dict_async("IMG_8888.jpeg", new_ann_json)
            run_coroutine(coroutine)
        """
        if type(ann) is not dict:
            raise TypeError("Ann should be a dict, not a {}".format(type(ann)))
        dst_ann_path = self.get_ann_path(item_name)
        os.makedirs(os.path.dirname(dst_ann_path), exist_ok=True)
        await dump_json_file_async(ann, dst_ann_path, indent=4)

    async def set_ann_async(self, item_name: str, ann: Annotation) -> None:
        """
        Replaces given annotation for given item name to dataset annotations directory in json format.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param ann: Annotation object.
        :type ann: :class:`Annotation<supervisely.annotation.annotation.Annotation>`
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely._utils import run_coroutine

            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            height, width = 500, 700
            new_ann = sly.Annotation((height, width))

            coroutine = ds.set_ann_async("IMG_0748.jpeg", new_ann)
            run_coroutine(coroutine)
        """
        if type(ann) is not self.annotation_class:
            raise TypeError(
                f"Type of 'ann' should be {self.annotation_class.__name__}, not a {type(ann).__name__}"
            )
        dst_ann_path = self.get_ann_path(item_name)
        await dump_json_file_async(ann.to_json(), dst_ann_path, indent=4)

    async def _add_ann_by_type_async(self, item_name, ann):
        """
        Add given annotation to dataset annotations dir and to dictionary items: item file name -> annotation file name
        :param item_name: str
        :param ann: Annotation class object, str, dict, None (generate exception error if param type is another)
        """
        # This is a new-style annotation name, so if there was no image with this name yet, there should not have been
        # an annotation either.
        self._item_to_ann[item_name] = item_name + ANN_EXT
        if ann is None:
            await self.set_ann_async(item_name, self._get_empty_annotaion(item_name))
        elif type(ann) is self.annotation_class:
            await self.set_ann_async(item_name, ann)
        elif type(ann) is str:
            await self.set_ann_file_async(item_name, ann)
        elif type(ann) is dict:
            await self.set_ann_dict_async(item_name, ann)
        else:
            raise TypeError("Unsupported type {!r} for ann argument".format(type(ann)))

    async def _add_item_file_async(
        self, item_name, item_path, _validate_item=True, _use_hardlink=False
    ):
        """
        Add given item file to dataset items directory. Generate exception error if item_name already exists in dataset
        or item name has unsupported extension
        :param item_name: str
        :param item_path: str
        :param _validate_item: bool
        :param _use_hardlink: bool
        """
        if item_path is None:
            return

        self._check_add_item_name(item_name)
        dst_item_path = os.path.join(self.item_dir, item_name)
        if (
            item_path != dst_item_path and item_path is not None
        ):  # used only for agent + api during download project + None to optimize internal usage
            hardlink_done = False
            if _use_hardlink:
                try:
                    loop = get_or_create_event_loop()
                    await loop.run_in_executor(None, os.link, item_path, dst_item_path)
                    hardlink_done = True
                except OSError:
                    pass
            if not hardlink_done:
                await copy_file_async(item_path, dst_item_path)
            if _validate_item:
                await self._validate_added_item_or_die_async(item_path)

    async def add_item_file_async(
        self,
        item_name: str,
        item_path: str,
        ann: Optional[Union[Annotation, str]] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
        item_info: Optional[Union[ImageInfo, Dict, str]] = None,
        img_info: Optional[Union[ImageInfo, Dict, str]] = None,
    ) -> None:
        """
        Adds given item file to dataset items directory, and adds given annotation to dataset annotations directory.
        If ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param item_path: Path to the item.
        :type item_path: :class:`str`
        :param ann: Annotation object or path to annotation json file.
        :type ann: :class:`Annotation<supervisely.annotation.annotation.Annotation>` or :class:`str`, optional
        :param _validate_item: Checks input files format.
        :type _validate_item: :class:`bool`, optional
        :param _use_hardlink: If True creates a hardlink pointing to src named dst, otherwise don't.
        :type _use_hardlink: :class:`bool`, optional
        :param item_info: ImageInfo object or ImageInfo object converted to dict or path to item info json file for copying to dataset item info directory.
        :type item_info: :class:`ImageInfo<supervisely.api.image_api.ImageInfo>` or :class:`dict` or :class:`str`, optional
        :param img_info: Deprecated version of item_info parameter. Can be removed in future versions.
        :type img_info: :class:`ImageInfo<supervisely.api.image_api.ImageInfo>` or :class:`dict` or :class:`str`, optional
        :return: None
        :rtype: NoneType
        :raises: :class:`RuntimeError` if item_name already exists in dataset or item name has unsupported extension.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            ds = sly.Dataset(dataset_path, sly.OpenMode.READ)

            ann = "/home/admin/work/supervisely/projects/lemons_annotated/ds1/ann/IMG_8888.jpeg.json"
            loop = sly.utils.get_or_create_event_loop()
            loop.run_until_complete(
                    ds.add_item_file_async("IMG_8888.jpeg", "/home/admin/work/supervisely/projects/lemons_annotated/ds1/img/IMG_8888.jpeg", ann=ann)
                )
            print(ds.item_exists("IMG_8888.jpeg"))
            # Output: True
        """
        # item_path is None when image is cached
        if item_path is None and ann is None and img_info is None:
            raise RuntimeError("No item_path or ann or img_info provided.")

        if item_info is not None and img_info is not None:
            raise RuntimeError(
                "At least one parameter of two (item_info and img_info) must be None."
            )

        if img_info is not None:
            logger.warning(
                "img_info parameter of add_item_file() method is deprecated and can be removed in future versions. Use item_info parameter instead."
            )
            item_info = img_info

        await self._add_item_file_async(
            item_name,
            item_path,
            _validate_item=_validate_item,
            _use_hardlink=_use_hardlink,
        )
        await self._add_ann_by_type_async(item_name, ann)
        await self._add_item_info_async(item_name, item_info)

    def to_coco(
        self,
        meta: ProjectMeta,
        return_type: Literal["path", "dict"] = "path",
        dest_dir: Optional[str] = None,
        copy_images: bool = False,
        with_captions=False,
        log_progress: bool = False,
        progress_cb: Optional[Callable] = None,
    ) -> Tuple[Dict, Union[None, Dict]]:
        """
        Convert Supervisely dataset to COCO format.

        Note:   Depending on the `return_type` and `with_captions` parameters, the function returns different values.
                If `return_type` is "path", the COCO annotation files will be saved to the disk.
                If `return_type` is "dict", the function returns COCO dataset in dictionary format.
                If `with_captions` is True, the function returns Tuple (instances and captions).

        :param meta: Project meta information.
        :type meta: :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`
        :param return_type: Return type (`path` or `dict`).
        :type return_type: :class:`str`, optional
        :param dest_dir: Path to save COCO dataset.
        :type dest_dir: :class:`str`, optional
        :param copy_images: If True, copies images to the COCO dataset directory.
        :type copy_images: :class:`bool`, optional
        :param with_captions: If True, returns captions
        :type with_captions: :class:`bool`, optional
        :param log_progress: If True, log progress.
        :type log_progress: :class:`str`, optional
        :param progress_cb: Progress callback.
        :type progress_cb: :class:`Callable`, optional
        :return: COCO dataset in dictionary format.
        :rtype: :class:`dict`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            project = sly.Project(project_path, sly.OpenMode.READ)

            for ds in project.datasets:
                dest_dir = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
                coco: Tuple[Dict, Dict] = ds.to_coco(project.meta, save=True, dest_dir=dest_dir)
        """

        from supervisely.convert import dataset_to_coco

        return dataset_to_coco(
            self,
            meta=meta,
            return_type=return_type,
            dest_dir=dest_dir,
            copy_images=copy_images,
            with_captions=with_captions,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    def to_yolo(
        self,
        meta: ProjectMeta,
        dest_dir: Optional[str] = None,
        task_type: Literal["detect", "segment", "pose"] = "detect",
        log_progress: bool = False,
        progress_cb: Optional[Callable] = None,
        is_val: Optional[bool] = None,
    ):
        """
        Convert Supervisely dataset to YOLO format.

        :param meta: Project meta information.
        :type meta: :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`
        :param dest_dir: Path to save YOLO dataset.
        :type dest_dir: :class:`str`, optional
        :param task_type: Task type.
        :type task_type: :class:`str`, optional
        :param log_progress: If True, log progress.
        :type log_progress: :class:`str`, optional
        :param progress_cb: Progress callback.
        :type progress_cb: :class:`Callable`, optional
        :param is_val: If True, the dataset is a validation dataset.
        :type is_val: :class:`bool`, optional
        :return: YOLO dataset in dictionary format.
        :rtype: :class:`dict`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            project = sly.Project(project_path, sly.OpenMode.READ)

            for ds in project.datasets:
                dest_dir = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
                ds.to_yolo(project.meta, dest_dir=dest_dir)
        """

        from supervisely.convert import dataset_to_yolo

        return dataset_to_yolo(
            self,
            meta=meta,
            dest_dir=dest_dir,
            task_type=task_type,
            log_progress=log_progress,
            progress_cb=progress_cb,
            is_val=is_val,
        )

    def to_pascal_voc(
        self,
        meta: ProjectMeta,
        dest_dir: Optional[str] = None,
        train_val_split_coef: float = 0.8,
        log_progress: bool = False,
        progress_cb: Optional[Union[Callable, tqdm]] = None,
    ) -> Tuple[Dict, Union[None, Dict]]:
        """
        Convert Supervisely dataset to Pascal VOC format.

        :param meta: Project meta information.
        :type meta: :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`
        :param dest_dir: Destination directory.
        :type dest_dir: :class:`str`, optional
        :param train_val_split_coef: Coefficient for splitting images into train and validation sets.
        :type train_val_split_coef: :class:`float`, optional
        :param log_progress: If True, log progress.
        :type log_progress: :class:`str`, optional
        :param progress_cb: Progress callback.
        :type progress_cb: :class:`Callable`, optional
        :return: None
        :rtype: NoneType

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            project = sly.Project(project_path, sly.OpenMode.READ)

            for ds in project.datasets:
                dest_dir = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
                ds.to_pascal_voc(project.meta, dest_dir=dest_dir)
        """
        from supervisely.convert import dataset_to_pascal_voc

        dataset_to_pascal_voc(
            self,
            meta=meta,
            dest_dir=dest_dir,
            train_val_split_coef=train_val_split_coef,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    def get_blob_img_bytes(self, image_name: str) -> bytes:
        """
        Get image bytes from blob file.

        :param image_name: Image name with extension.
        :type image_name: :class:`str`
        :return: Bytes of the image.
        :rtype: :class:`bytes`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/path/to/project/lemons_annotated/ds1"
            dataset = sly.Dataset(dataset_path, sly.OpenMode.READ)
            image_name = "IMG_0748.jpeg"

            img_bytes = dataset.get_blob_img_bytes(image_name)
        """

        if self.project_dir is None:
            raise RuntimeError("Project directory is not set. Cannot get blob image bytes.")

        blob_image_info = None

        for offset in self.blob_offsets:
            for batch in BlobImageInfo.load_from_pickle_generator(offset):
                for file in batch:
                    if file.name == image_name:
                        blob_image_info = file
                        blob_file_name = removesuffix(Path(offset).name, OFFSETS_PKL_SUFFIX)
                        break
        if blob_image_info is None:
            logger.debug(
                f"Image '{image_name}' not found in blob offsets. "
                f"Make sure that the image is stored in the blob file."
            )
            return None

        blob_file_path = os.path.join(self.project_dir, self.blob_dir_name, blob_file_name + ".tar")
        if file_exists(blob_file_path):
            with open(blob_file_path, "rb") as f:
                f.seek(blob_image_info.offset_start)
                img_bytes = f.read(blob_image_info.offset_end - blob_image_info.offset_start)
        else:
            logger.debug(
                f"Blob file '{blob_file_path}' not found. "
                f"Make sure that the blob file exists in the specified directory."
            )
            img_bytes = None
        return img_bytes

    def get_blob_img_np(self, image_name: str) -> np.ndarray:
        """
        Get image as numpy array from blob file.

        :param image_name: Image name with extension.
        :type image_name: :class:`str`
        :return: Numpy array of the image.
        :rtype: :class:`numpy.ndarray`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/path/to/project/lemons_annotated/ds1"
            dataset = sly.Dataset(dataset_path, sly.OpenMode.READ)
            image_name = "IMG_0748.jpeg"

            img_np = dataset.get_blob_img_np(image_name)
        """
        img_bytes = self.get_blob_img_bytes(image_name)
        if img_bytes is None:
            return None
        return sly_image.read_bytes(img_bytes)


class Project:
    """
    Project is a parent directory for dataset. Project object is immutable.

    :param directory: Path to project directory.
    :type directory: :class:`str`
    :param mode: Determines working mode for the given project.
    :type mode: :class:`OpenMode<OpenMode>`
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        project_path = "/home/admin/work/supervisely/projects/lemons_annotated"
        project = sly.Project(project_path, sly.OpenMode.READ)
    """

    dataset_class = Dataset
    blob_dir_name = "blob"

    class DatasetDict(KeyIndexedCollection):
        """
        :class:`Datasets<Dataset>` collection of :class:`Project<Project>`.
        """

        item_type = Dataset

        def __next__(self):
            for dataset in self.items():
                yield dataset

        def items(self) -> List[KeyObject]:
            return sorted(self._collection.values(), key=lambda x: x.parents)

    def __init__(
        self,
        directory: str,
        mode: Optional[OpenMode] = None,
        project_id: Optional[int] = None,
        api: Optional[sly.Api] = None,
    ):
        if project_id is not None:
            raise NotImplementedError(
                "Opening project from the API is not implemented yet. Please use local mode "
                "by providing directory and mode parameters. "
                "This feature will be implemented later."
            )
        if mode is None and project_id is None:
            raise ValueError("One of the parameters 'mode' or 'project_id' should be set.")
        if type(mode) is not OpenMode and mode is not None:
            raise TypeError(
                "Argument 'mode' has type {!r}. Correct type is OpenMode".format(type(mode))
            )
        if project_id is not None and api is None:
            raise ValueError("Parameter 'api' should be set if 'project_id' is set.")

        parent_dir, name = Project._parse_path(directory)
        self._parent_dir = parent_dir
        self._blob_dir = os.path.join(directory, self.blob_dir_name)
        self._api = api
        self.project_id = project_id

        if project_id is not None:
            self._info = api.project.get_info_by_id(project_id)
            self._name = self._info.name
        else:
            self._info = None
            self._name = name
        self._datasets = Project.DatasetDict()  # ds_name -> dataset object
        self._meta = None
        self._blob_files = []
        if project_id is not None:
            self._read_api()
        elif mode is OpenMode.READ:
            self._read()
        else:
            self._create()

    @staticmethod
    def get_url(id: int) -> str:
        """
        Get URL to datasets list in Supervisely.

        :param id: :class:`Project<Project>` ID in Supervisely.
        :type id: :class:`int`
        :return: URL to datasets list.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            from supervisely import Project

            project_id = 10093
            datasets_link = Project.get_url(project_id)

            print(datasets_link)
            # Output: "/projects/10093/datasets"
        """
        res = f"/projects/{id}/datasets"
        if is_development():
            res = abs_url(res)
        return res

    @property
    def parent_dir(self) -> str:
        """
        Project parent directory.

        :return: Path to project parent directory
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            print(project.parent_dir)
            # Output: '/home/admin/work/supervisely/projects'
        """
        return self._parent_dir

    @property
    def blob_dir(self) -> str:
        """
        Directory for project blobs.
        Blobs are .tar files with images. Used for fast data transfer.

        :return: Path to project blob directory
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            print(project.blob_dir)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/blob'
        """
        return self._blob_dir

    @property
    def name(self) -> str:
        """
        Project name.

        :return: Project name.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            print(project.name)
            # Output: 'lemons_annotated'
        """
        return self._name

    @property
    def type(self) -> str:
        """
        Project type.

        :return: Project type.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            print(project.type)
            # Output: 'images'
        """
        return ProjectType.IMAGES.value

    @property
    def datasets(self) -> Project.DatasetDict:
        """
        Project datasets.

        :return: Datasets
        :rtype: :class:`DatasetDict<supervisely.project.project.Project.DatasetDict>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            for dataset in project.datasets:
                print(dataset.name)
                # Output: ds1
                #         ds2
        """

        return self._datasets

    @property
    def meta(self) -> ProjectMeta:
        """
        Project meta.

        :return: ProjectMeta object
        :rtype: :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            print(project.meta)
            # Output:
            # +-------+--------+----------------+--------+
            # |  Name | Shape  |     Color      | Hotkey |
            # +-------+--------+----------------+--------+
            # |  kiwi | Bitmap |  [255, 0, 0]   |        |
            # | lemon | Bitmap | [81, 198, 170] |        |
            # +-------+--------+----------------+--------+
            # Tags
            # +------+------------+-----------------+--------+---------------+--------------------+
            # | Name | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
            # +------+------------+-----------------+--------+---------------+--------------------+
        """
        return self._meta

    @property
    def directory(self) -> str:
        """
        Path to the project directory.

        :return: Path to the project directory
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            print(project.directory)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated'
        """
        return os.path.join(self.parent_dir, self.name)

    @property
    def total_items(self) -> int:
        """
        Total number of items in project.

        :return: Total number of items in project
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            print(project.total_items)
            # Output: 12
        """
        return sum(len(ds) for ds in self._datasets)

    @property
    def blob_files(self) -> List[str]:
        """
        List of blob files.

        :return: List of blob files
        :rtype: :class:`list`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            print(project.blob_files)
            # Output: []
        """
        return self._blob_files

    @blob_files.setter
    def blob_files(self, blob_files: List[str]) -> None:
        """
        Sets blob files to the project.

        :param blob_files: List of blob files.
        :type
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            project.blob_files = ["blob_file.tar"]
        """
        self._blob_files = blob_files

    def add_blob_file(self, file_name: str) -> None:
        """
        Adds blob file to the project.

        :param file_name: File name.
        :type file_name: :class:`str`
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            project.add_blob_file("blob_file.tar")
        """
        self._blob_files.append(file_name)

    def get_classes_stats(
        self,
        dataset_names: Optional[List[str]] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        result = {}
        for ds in self.datasets:
            ds: Dataset
            if dataset_names is not None and ds.name not in dataset_names:
                continue
            ds_stats = ds.get_classes_stats(
                self.meta,
                return_objects_count,
                return_figures_count,
                return_items_count,
            )
            for stat_name, classes_stats in ds_stats.items():
                if stat_name not in result.keys():
                    result[stat_name] = {}
                for class_name, class_count in classes_stats.items():
                    if class_name not in result[stat_name].keys():
                        result[stat_name][class_name] = 0
                    result[stat_name][class_name] += class_count

        return result

    def _get_project_meta_path(self):
        """
        :return: str (path to project meta file(meta.json))
        """
        return os.path.join(self.directory, "meta.json")

    def _read(self):
        meta_json = load_json_file(self._get_project_meta_path())
        self._meta = ProjectMeta.from_json(meta_json)
        if dir_exists(self.blob_dir):
            self.blob_files = [Path(file).name for file in list_files(self.blob_dir)]
        else:
            self.blob_files = []

        ignore_dirs = self.dataset_class.ignorable_dirs()  # dir names that can not be datasets

        ignore_content_dirs = ignore_dirs.copy()  # dir names which can not contain datasets
        ignore_content_dirs.pop(ignore_content_dirs.index(self.dataset_class.datasets_dir()))

        possible_datasets = subdirs_tree(self.directory, ignore_dirs, ignore_content_dirs)

        for ds_name in possible_datasets:
            parents = ds_name.split(os.path.sep)
            parents = [p for p in parents if p != self.dataset_class.datasets_dir()]
            if len(parents) > 1:
                parents.pop(-1)
            else:
                parents = None
            try:
                current_dataset = self.dataset_class(
                    os.path.join(self.directory, ds_name),
                    OpenMode.READ,
                    parents=parents,
                )
                if current_dataset.name not in self._datasets._collection:
                    self._datasets = self._datasets.add(current_dataset)
                else:
                    logger.debug(
                        f"Dataset '{current_dataset.name}' already exists in project '{self.name}'. Skip adding to collection."
                    )
            except Exception as ex:
                logger.warning(ex)

        if self.total_items == 0:
            raise RuntimeError("Project is empty")

    def _read_api(self):
        self._meta = ProjectMeta.from_json(self._api.project.get_meta(self.project_id))
        for parents, dataset_info in self._api.dataset.tree(self.project_id):
            relative_path = self.dataset_class._get_dataset_path(dataset_info.name, parents)
            dataset_path = os.path.join(self.directory, relative_path)
            current_dataset = self.dataset_class(
                dataset_path, parents=parents, dataset_id=dataset_info.id, api=self._api
            )
            self._datasets = self._datasets.add(current_dataset)

    def _create(self):
        if dir_exists(self.directory):
            if len(list_files_recursively(self.directory)) > 0:
                raise RuntimeError(
                    "Cannot create new project {!r}. Directory {!r} already exists and is not empty".format(
                        self.name, self.directory
                    )
                )
        else:
            mkdir(self.directory)
        self.set_meta(ProjectMeta())
        self.blob_files = []

    def validate(self):
        # @TODO: remove?
        pass

    def set_meta(self, new_meta: ProjectMeta) -> None:
        """
        Saves given :class:`meta<supervisely.project.project_meta.ProjectMeta>` to project directory in json format.

        :param new_meta: ProjectMeta object.
        :type new_meta: :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            proj_lemons = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            proj_kiwi = sly.Project("/home/admin/work/supervisely/projects/kiwi_annotated", sly.OpenMode.READ)

            proj_lemons.set_meta(proj_kiwi.meta)

            print(project.proj_lemons)
            # Output:
            # +-------+--------+----------------+--------+
            # |  Name | Shape  |     Color      | Hotkey |
            # +-------+--------+----------------+--------+
            # |  kiwi | Bitmap |  [255, 0, 0]   |        |
            # +-------+--------+----------------+--------+
        """
        self._meta = new_meta
        dump_json_file(self.meta.to_json(), self._get_project_meta_path(), indent=4)

    def __iter__(self):
        return next(self)

    def __next__(self):
        for dataset in self._datasets:
            yield dataset

    def create_dataset(self, ds_name: str, ds_path: Optional[str] = None) -> Dataset:
        """
        Creates a subdirectory with given name and all intermediate subdirectories for items and annotations in project directory, and also adds created dataset
        to the collection of all datasets in the project.

        :param ds_name: Dataset name.
        :type ds_name: :class:`str`
        :return: Dataset object
        :rtype: :class:`Dataset<Dataset>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)

            for dataset in project.datasets:
                print(dataset.name)

            # Output: ds1
            #         ds2

            project.create_dataset("ds3")

            for dataset in project.datasets:
                print(dataset.name)

            # Output: ds1
            #         ds2
            #         ds3
        """
        if ds_path is None:
            ds_path = os.path.join(self.directory, ds_name)
        else:
            ds_path = os.path.join(self.directory, ds_path)

        ds = self.dataset_class(ds_path, OpenMode.CREATE)
        self._datasets = self._datasets.add(ds)
        return ds

    def copy_data(
        self,
        dst_directory: str,
        dst_name: Optional[str] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
    ) -> Project:
        """
        Makes a copy of the :class:`Project<Project>`.

        :param dst_directory: Path to project parent directory.
        :type dst_directory: :class:`str`
        :param dst_name: Project name.
        :type dst_name: :class:`str`, optional
        :param _validate_item: Checks input files format.
        :type _validate_item: :class:`bool`, optional
        :param _use_hardlink: If True creates a hardlink pointing to src named dst, otherwise don't.
        :type _use_hardlink: :class:`bool`, optional
        :return: Project object.
        :rtype: :class:`Project<Project>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            print(project.total_items)
            # Output: 6

            new_project = project.copy_data("/home/admin/work/supervisely/projects/", "lemons_copy")
            print(new_project.total_items)
            # Output: 6
        """
        dst_name = dst_name if dst_name is not None else self.name
        new_project = Project(os.path.join(dst_directory, dst_name), OpenMode.CREATE)
        new_project.set_meta(self.meta)

        for ds in self:
            new_ds = new_project.create_dataset(ds.name)

            for item_name in ds:
                item_path, ann_path = ds.get_item_paths(item_name)
                item_info_path = ds.get_item_info_path(item_name)

                item_path = item_path if os.path.isfile(item_path) else None
                ann_path = ann_path if os.path.isfile(ann_path) else None
                item_info_path = item_info_path if os.path.isfile(item_info_path) else None

                new_ds.add_item_file(
                    item_name,
                    item_path,
                    ann_path,
                    _validate_item=_validate_item,
                    _use_hardlink=_use_hardlink,
                    item_info=item_info_path,
                )
        return new_project

    @staticmethod
    def _parse_path(project_dir):
        """
        Split given path to project on parent directory and directory where project is located
        :param project_dir: str
        :return: str, str
        """
        # alternative implementation
        # temp_parent_dir = os.path.dirname(parent_dir)
        # temp_name = os.path.basename(parent_dir)

        parent_dir, pr_name = os.path.split(project_dir.rstrip("/"))
        if not pr_name:
            raise RuntimeError("Unable to determine project name.")
        return parent_dir, pr_name

    @staticmethod
    def to_segmentation_task(
        src_project_dir: str,
        dst_project_dir: Optional[str] = None,
        inplace: Optional[bool] = False,
        target_classes: Optional[List[str]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        segmentation_type: Optional[str] = "semantic",
        bg_name: Optional[str] = "__bg__",
        bg_color: Optional[List[int]] = None,
    ) -> None:
        """
        Makes a copy of the :class:`Project<Project>`, converts annotations to
        :class:`Bitmaps<supervisely.geometry.bitmap.Bitmap>` and updates
        :class:`project meta<supervisely.project.project_meta.ProjectMeta>`.

        You will able to get item's segmentation masks location by :class:`dataset.get_seg_path(item_name)<supervisely.project.project.Dataset.get_seg_path>` method.

        :param src_project_dir: Path to source project directory.
        :type src_project_dir: :class:`str`
        :param dst_project_dir: Path to destination project directory. Must be None If inplace=True.
        :type dst_project_dir: :class:`str`, optional
        :param inplace: Modifies source project If True. Must be False If dst_project_dir is specified.
        :type inplace: :class:`bool`, optional
        :param target_classes: Classes list to include to destination project. If segmentation_type="semantic",
                               background class will be added automatically (by default "__bg__").
        :type target_classes: :class:`list` [ :class:`str` ], optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param segmentation_type: One of: {"semantic", "instance"}. If segmentation_type="semantic", background class
                                  will be added automatically (by default "__bg__") and instances will be converted to non overlapping semantic segmentation mask.
        :type segmentation_type: :class:`str`
        :param bg_name: Default background class name, used for semantic segmentation.
        :type bg_name: :class:`str`, optional
        :param bg_color: Default background class color, used for semantic segmentation.
        :type bg_color: :class:`list`, optional. Default is [0, 0, 0]
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            source_project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            seg_project_path = "/home/admin/work/supervisely/projects/lemons_segmentation"
            sly.Project.to_segmentation_task(
                src_project_dir=source_project.directory,
                dst_project_dir=seg_project_path
            )
            seg_project = sly.Project(seg_project_path, sly.OpenMode.READ)
        """

        _bg_class_name = bg_name
        bg_color = bg_color or [0, 0, 0]
        _bg_obj_class = ObjClass(_bg_class_name, Bitmap, color=bg_color)

        if dst_project_dir is None and inplace is False:
            raise ValueError(
                f"Original project in folder {src_project_dir} will be modified. Please, set 'inplace' "
                f"argument (inplace=True) directly"
            )
        if inplace is True and dst_project_dir is not None:
            raise ValueError("dst_project_dir has to be None if inplace is True")

        if dst_project_dir is not None:
            if not dir_exists(dst_project_dir):
                mkdir(dst_project_dir)
            elif not dir_empty(dst_project_dir):
                raise ValueError(f"Destination directory {dst_project_dir} is not empty")

        src_project = Project(src_project_dir, OpenMode.READ)
        dst_meta = src_project.meta.clone()

        dst_meta, dst_mapping = dst_meta.to_segmentation_task(target_classes=target_classes)

        if segmentation_type == "semantic" and dst_meta.obj_classes.get(_bg_class_name) is None:
            dst_meta = dst_meta.add_obj_class(_bg_obj_class)

        if target_classes is not None:
            if segmentation_type == "semantic":
                if _bg_class_name not in target_classes:
                    target_classes.append(_bg_class_name)

            # check that all target classes are in destination project meta
            for class_name in target_classes:
                if dst_meta.obj_classes.get(class_name) is None:
                    raise KeyError(f"Class {class_name} not found in destination project meta")

            dst_meta = dst_meta.clone(
                obj_classes=ObjClassCollection(
                    [dst_meta.obj_classes.get(class_name) for class_name in target_classes]
                )
            )

        if inplace is False:
            dst_project = Project(dst_project_dir, OpenMode.CREATE)
            dst_project.set_meta(dst_meta)

        for src_dataset in src_project.datasets:
            if inplace is False:
                dst_dataset = dst_project.create_dataset(src_dataset.name)

            for item_name in src_dataset:
                img_path, ann_path = src_dataset.get_item_paths(item_name)
                ann = Annotation.load_json_file(ann_path, src_project.meta)

                if segmentation_type == "semantic":
                    seg_ann = ann.add_bg_object(_bg_obj_class)

                    dst_mapping[_bg_obj_class] = _bg_obj_class
                    seg_ann = seg_ann.to_nonoverlapping_masks(dst_mapping)  # get_labels with bg

                    seg_ann = seg_ann.to_segmentation_task()
                elif segmentation_type == "instance":
                    seg_ann = ann.to_nonoverlapping_masks(
                        dst_mapping
                    )  # rendered instances and filter classes
                elif segmentation_type == "panoptic":
                    raise NotImplementedError

                seg_path = None
                if inplace is False:
                    if file_exists(img_path):
                        dst_dataset.add_item_file(item_name, img_path, seg_ann)
                    else:
                        # if local project has no images
                        dst_dataset._add_ann_by_type(item_name, seg_ann)
                    seg_path = dst_dataset.get_seg_path(item_name)
                else:
                    # replace existing annotation
                    src_dataset.set_ann(item_name, seg_ann)
                    seg_path = src_dataset.get_seg_path(item_name)

                # save rendered segmentation
                # seg_ann.to_indexed_color_mask(seg_path, palette=palette["colors"], colors=len(palette["names"]))
                seg_ann.to_indexed_color_mask(seg_path)
                if progress_cb is not None:
                    progress_cb(1)

        if inplace is True:
            src_project.set_meta(dst_meta)

    @staticmethod
    def to_detection_task(
        src_project_dir: str,
        dst_project_dir: Optional[str] = None,
        inplace: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Makes a copy of the :class:`Project<Project>`, converts annotations to
        :class:`Rectangles<supervisely.geometry.rectangle.Rectangle>` and updates
        :class:`project meta<supervisely.project.project_meta.ProjectMeta>`.

        :param src_project_dir: Path to source project directory.
        :type src_project_dir: :class:`str`
        :param dst_project_dir: Path to destination project directory. Must be None If inplace=True.
        :type dst_project_dir: :class:`str`, optional
        :param inplace: Modifies source project If True. Must be False If dst_project_dir is specified.
        :type inplace: :class:`bool`, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            source_project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
            det_project_path = "/home/admin/work/supervisely/projects/lemons_detection"
            sly.Project.to_detection_task(
                src_project_dir=source_project.directory,
                dst_project_dir=det_project_path
            )
            det_project = sly.Project(det_project_path, sly.OpenMode.READ)
        """
        if dst_project_dir is None and inplace is False:
            raise ValueError(
                f"Original project in folder {src_project_dir} will be modified. Please, set 'inplace' "
                f"argument (inplace=True) directly"
            )
        if inplace is True and dst_project_dir is not None:
            raise ValueError("dst_project_dir has to be None if inplace is True")

        if dst_project_dir is not None:
            if not dir_exists(dst_project_dir):
                mkdir(dst_project_dir)
            elif not dir_empty(dst_project_dir):
                raise ValueError(f"Destination directory {dst_project_dir} is not empty")

        src_project = Project(src_project_dir, OpenMode.READ)
        det_meta, det_mapping = src_project.meta.to_detection_task(convert_classes=True)

        if inplace is False:
            dst_project = Project(dst_project_dir, OpenMode.CREATE)
            dst_project.set_meta(det_meta)

        for src_dataset in src_project.datasets:
            if inplace is False:
                dst_dataset = dst_project.create_dataset(src_dataset.name)
            for item_name in src_dataset:
                img_path, ann_path = src_dataset.get_item_paths(item_name)
                ann = Annotation.load_json_file(ann_path, src_project.meta)
                det_ann = ann.to_detection_task(det_mapping)

                if inplace is False:
                    dst_dataset.add_item_file(item_name, img_path, det_ann)
                else:
                    # replace existing annotation
                    src_dataset.set_ann(item_name, det_ann)
                if progress_cb is not None:
                    progress_cb(1)

        if inplace is True:
            src_project.set_meta(det_meta)

    @staticmethod
    def remove_classes_except(
        project_dir: str,
        classes_to_keep: Optional[List[str]] = None,
        inplace: Optional[bool] = False,
    ) -> None:
        """
        Removes classes from Project with the exception of some classes.

        :param project_dir: Path to project directory.
        :type project_dir: :class:`str`
        :param classes_to_keep: Classes to keep in project.
        :type classes_to_keep: :class:`list` [ :class:`str` ], optional
        :param inplace: Checkbox that determines whether to change the source data in project or not.
        :type inplace: :class:`bool`, optional
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project(project_path, sly.OpenMode.READ)
            project.remove_classes_except(project_path, inplace=True)
        """
        if classes_to_keep is None:
            classes_to_keep = []
        classes_to_remove = []
        project = Project(project_dir, OpenMode.READ)
        for obj_class in project.meta.obj_classes:
            if obj_class.name not in classes_to_keep:
                classes_to_remove.append(obj_class.name)
        Project.remove_classes(project_dir, classes_to_remove, inplace)

    @staticmethod
    def remove_classes(
        project_dir: str,
        classes_to_remove: Optional[List[str]] = None,
        inplace: Optional[bool] = False,
    ) -> None:
        """
        Removes given classes from Project.

        :param project_dir: Path to project directory.
        :type project_dir: :class:`str`
        :param classes_to_remove: Classes to remove.
        :type classes_to_remove: :class:`list` [ :class:`str` ], optional
        :param inplace: Checkbox that determines whether to change the source data in project or not.
        :type inplace: :class:`bool`, optional
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.Project(project_path, sly.OpenMode.READ)
            classes_to_remove = ['lemon']
            project.remove_classes(project_path, classes_to_remove, inplace=True)
        """
        if classes_to_remove is None:
            classes_to_remove = []
        if inplace is False:
            raise ValueError(
                "Original data will be modified. Please, set 'inplace' argument (inplace=True) directly"
            )
        project = Project(project_dir, OpenMode.READ)
        for dataset in project.datasets:
            for item_name in dataset:
                img_path, ann_path = dataset.get_item_paths(item_name)
                ann = Annotation.load_json_file(ann_path, project.meta)
                new_labels = []
                for label in ann.labels:
                    if label.obj_class.name not in classes_to_remove:
                        new_labels.append(label)
                new_ann = ann.clone(labels=new_labels)
                dataset.set_ann(item_name, new_ann)
        new_classes = []
        for obj_class in project.meta.obj_classes:
            if obj_class.name not in classes_to_remove:
                new_classes.append(obj_class)
        new_meta = project.meta.clone(obj_classes=ObjClassCollection(new_classes))
        project.set_meta(new_meta)

    @staticmethod
    def _remove_items(
        project_dir,
        without_objects=False,
        without_tags=False,
        without_objects_and_tags=False,
        inplace=False,
    ):
        if inplace is False:
            raise ValueError(
                "Original data will be modified. Please, set 'inplace' argument (inplace=True) directly"
            )
        if without_objects is False and without_tags is False and without_objects_and_tags is False:
            raise ValueError(
                "One of the flags (without_objects / without_tags or without_objects_and_tags) have to be defined"
            )
        project = Project(project_dir, OpenMode.READ)
        for dataset in project.datasets:
            items_to_delete = []
            for item_name in dataset:
                img_path, ann_path = dataset.get_item_paths(item_name)
                ann = Annotation.load_json_file(ann_path, project.meta)
                if (
                    (without_objects and len(ann.labels) == 0)
                    or (without_tags and len(ann.img_tags) == 0)
                    or (without_objects_and_tags and ann.is_empty())
                ):
                    items_to_delete.append(item_name)
            for item_name in items_to_delete:
                dataset.delete_item(item_name)

    @staticmethod
    def remove_items_without_objects(project_dir: str, inplace: Optional[bool] = False) -> None:
        """
        Remove items(images and annotations) without objects from Project.

        :param project_dir: Path to project directory.
        :type project_dir: :class:`str`
        :param inplace: Checkbox that determines whether to change the source data in project or not.
        :type inplace: :class:`bool`, optional
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            sly.Project.remove_items_without_objects(project_path, inplace=True)
        """
        Project._remove_items(project_dir=project_dir, without_objects=True, inplace=inplace)

    @staticmethod
    def remove_items_without_tags(project_dir: str, inplace: Optional[bool] = False) -> None:
        """
        Remove items(images and annotations) without tags from Project.

        :param project_dir: Path to project directory.
        :type project_dir: :class:`str`
        :param inplace: Checkbox that determines whether to change the source data in project or not.
        :type inplace: :class:`bool`, optional
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            sly.Project.remove_items_without_tags(project_path, inplace=True)
        """
        Project._remove_items(project_dir=project_dir, without_tags=True, inplace=inplace)

    @staticmethod
    def remove_items_without_both_objects_and_tags(
        project_dir: str, inplace: Optional[bool] = False
    ) -> None:
        """
        Remove items(images and annotations) without objects and tags from Project.

        :param project_dir: Path to project directory.
        :type project_dir: :class:`str`
        :param inplace: Checkbox that determines whether to change the source data in project or not.
        :type inplace: :class:`bool`, optional
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            sly.Project.remove_items_without_both_objects_and_tags(project_path, inplace=True)
        """
        Project._remove_items(
            project_dir=project_dir, without_objects_and_tags=True, inplace=inplace
        )

    def get_item_paths(self, item_name) -> ItemPaths:
        # TODO: remove?
        raise NotImplementedError("Method available only for dataset")

    @staticmethod
    def get_train_val_splits_by_count(
        project_dir: str, train_count: int, val_count: int
    ) -> Tuple[List[ItemInfo], List[ItemInfo]]:
        """
        Get train and val items information from project by given train and val counts.

        :param project_dir: Path to project directory.
        :type project_dir: :class:`str`
        :param train_count: Number of train items.
        :type train_count: :class:`int`
        :param val_count: Number of val items.
        :type val_count: :class:`int`
        :raises: :class:`ValueError` if total_count != train_count + val_count
        :return: Tuple with lists of train items information and val items information
        :rtype: :class:`list` [ :class:`ItemInfo<ItemInfo>` ], :class:`list` [ :class:`ItemInfo<ItemInfo>` ]
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            train_count = 4
            val_count = 2
            train_items, val_items = sly.Project.get_train_val_splits_by_count(
                project_path,
                train_count,
                val_count
            )
        """

        def _list_items_for_splits(project) -> List[ItemInfo]:
            items = []
            for dataset in project.datasets:
                for item_name in dataset:
                    items.append(
                        ItemInfo(
                            dataset_name=dataset.name,
                            name=item_name,
                            img_path=dataset.get_img_path(item_name),
                            ann_path=dataset.get_ann_path(item_name),
                        )
                    )
            return items

        project = Project(project_dir, OpenMode.READ)
        if project.total_items != train_count + val_count:
            raise ValueError("total_count != train_count + val_count")
        all_items = _list_items_for_splits(project)
        random.shuffle(all_items)
        train_items = all_items[:train_count]
        val_items = all_items[train_count:]
        return train_items, val_items

    @staticmethod
    def get_train_val_splits_by_tag(
        project_dir: str,
        train_tag_name: str,
        val_tag_name: str,
        untagged: Optional[str] = "ignore",
    ) -> Tuple[List[ItemInfo], List[ItemInfo]]:
        """
        Get train and val items information from project by given train and val tags names.

        :param project_dir: Path to project directory.
        :type project_dir: :class:`str`
        :param train_tag_name: Train tag name.
        :type train_tag_name: :class:`str`
        :param val_tag_name: Val tag name.
        :type val_tag_name: :class:`str`
        :param untagged: Actions in case of absence of train_tag_name and val_tag_name in project.
        :type untagged: :class:`str`, optional
        :raises: :class:`ValueError` if untagged not in ["ignore", "train", "val"]
        :return: Tuple with lists of train items information and val items information
        :rtype: :class:`list` [ :class:`ItemInfo<ItemInfo>` ], :class:`list` [ :class:`ItemInfo<ItemInfo>` ]
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            train_tag_name = 'train'
            val_tag_name = 'val'
            train_items, val_items = sly.Project.get_train_val_splits_by_tag(
                project_path,
                train_tag_name,
                val_tag_name
            )
        """
        untagged_actions = ["ignore", "train", "val"]
        if untagged not in untagged_actions:
            raise ValueError(
                f"Unknown untagged action {untagged}. Should be one of {untagged_actions}"
            )
        project = Project(project_dir, OpenMode.READ)
        train_items = []
        val_items = []
        for dataset in project.datasets:
            for item_name in dataset:
                img_path, ann_path = dataset.get_item_paths(item_name)
                info = ItemInfo(dataset.name, item_name, img_path, ann_path)

                ann = Annotation.load_json_file(ann_path, project.meta)
                if ann.img_tags.get(train_tag_name) is not None:
                    train_items.append(info)
                if ann.img_tags.get(val_tag_name) is not None:
                    val_items.append(info)
                if (
                    ann.img_tags.get(train_tag_name) is None
                    and ann.img_tags.get(val_tag_name) is None
                ):
                    # untagged item
                    if untagged == "ignore":
                        continue
                    elif untagged == "train":
                        train_items.append(info)
                    elif untagged == "val":
                        val_items.append(info)
        return train_items, val_items

    @staticmethod
    def get_train_val_splits_by_dataset(
        project_dir: str, train_datasets: List[str], val_datasets: List[str]
    ) -> Tuple[List[ItemInfo], List[ItemInfo]]:
        """
        Get train and val items information from project by given train and val datasets names.

        :param project_dir: Path to project directory.
        :type project_dir: :class:`str`
        :param train_datasets: List of train datasets names.
        :type train_datasets: :class:`list` [ :class:`str` ]
        :param val_datasets: List of val datasets names.
        :type val_datasets: :class:`list` [ :class:`str` ]
        :raises: :class:`KeyError` if dataset name not found in project
        :return: Tuple with lists of train items information and val items information
        :rtype: :class:`list` [ :class:`ItemInfo<ItemInfo>` ], :class:`list` [ :class:`ItemInfo<ItemInfo>` ]
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            train_datasets = ['ds1', 'ds2']
            val_datasets = ['ds3', 'ds4']
            train_items, val_items = sly.Project.get_train_val_splits_by_dataset(
                project_path,
                train_datasets,
                val_datasets
            )
        """

        def _add_items_to_list(project, datasets_names, items_list):
            for dataset_name in datasets_names:
                dataset = project.datasets.get(dataset_name)
                if dataset is None:
                    raise KeyError(f"Dataset '{dataset_name}' not found")
                for item_name in dataset:
                    img_path, ann_path = dataset.get_item_paths(item_name)
                    info = ItemInfo(dataset.name, item_name, img_path, ann_path)
                    items_list.append(info)

        project = Project(project_dir, OpenMode.READ)
        train_items = []
        _add_items_to_list(project, train_datasets, train_items)
        val_items = []
        _add_items_to_list(project, val_datasets, val_items)
        return train_items, val_items

    @staticmethod
    def get_train_val_splits_by_collections(
        project_dir: str,
        train_collections: List[int],
        val_collections: List[int],
        project_id: int,
        api: Api,
    ) -> Tuple[List[ItemInfo], List[ItemInfo]]:
        """
        Get train and val items information from project by given train and val collections IDs.

        :param project_dir: Path to project directory.
        :type project_dir: :class:`str`
        :param train_collections: List of train collections IDs.
        :type train_collections: :class:`list` [ :class:`int` ]
        :param val_collections: List of val collections IDs.
        :type val_collections: :class:`list` [ :class:`int` ]
        :param project_id: Project ID.
        :type project_id: :class:`int`
        :param api: Supervisely API address and token.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :raises: :class:`KeyError` if collection ID not found in project
        :return: Tuple with lists of train items information and val items information
        :rtype: :class:`list` [ :class:`ItemInfo<ItemInfo>` ], :class:`list` [ :class:`ItemInfo<ItemInfo>` ]
        """
        from supervisely.api.entities_collection_api import CollectionTypeFilter

        project = Project(project_dir, OpenMode.READ)

        ds_id_to_name = {}
        for parents, ds_info in api.dataset.tree(project_id):
            full_name = "/".join(parents + [ds_info.name])
            ds_id_to_name[ds_info.id] = full_name

        train_items = []
        val_items = []

        for collection_ids, items_dict in [
            (train_collections, train_items),
            (val_collections, val_items),
        ]:
            for collection_id in collection_ids:
                collection_items = api.entities_collection.get_items(
                    collection_id=collection_id,
                    project_id=project_id,
                    collection_type=CollectionTypeFilter.DEFAULT,
                )
                for item in collection_items:
                    ds_name = ds_id_to_name.get(item.dataset_id)
                    ds = project.datasets.get(ds_name)
                    img_path, ann_path = ds.get_item_paths(item.name)
                    info = ItemInfo(ds_name, item.name, img_path, ann_path)
                    items_dict.append(info)

        return train_items, val_items

    @staticmethod
    def download(
        api: Api,
        project_id: int,
        dest_dir: str,
        dataset_ids: Optional[List[int]] = None,
        log_progress: bool = True,
        batch_size: Optional[int] = 50,
        cache: Optional[FileCache] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        only_image_tags: Optional[bool] = False,
        save_image_info: Optional[bool] = False,
        save_images: bool = True,
        save_image_meta: bool = False,
        resume_download: bool = False,
        **kwargs,
    ) -> None:
        """
        Download project from Supervisely to the given directory.

        :param api: Supervisely API address and token.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param project_id: Supervisely downloadable project ID.
        :type project_id: :class:`int`
        :param dest_dir: Destination directory.
        :type dest_dir: :class:`str`
        :param dataset_ids: Dataset IDs.
        :type dataset_ids: :class:`list` [ :class:`int` ], optional
        :param log_progress: Show uploading progress bar.
        :type log_progress: :class:`bool`
        :param batch_size: The number of images in the batch when they are loaded to a host.
        :type batch_size: :class:`int`, optional
        :param cache: FileCache object.
        :type cache: :class:`FileCache<supervisely.io.fs_cache.FileCache>`, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param only_image_tags: Download project with only images tags (without objects tags).
        :type only_image_tags: :class:`bool`, optional
        :param save_image_info: Download images infos or not.
        :type save_image_info: :class:`bool`, optional
        :param save_images: Download images or not.
        :type save_images: :class:`bool`, optional
        :param save_image_meta: Download images metadata in JSON format or not.
        :type save_image_meta: :class:`bool`, optional
        :param download_blob_files: Default is False. It will download images in classic way.
                                If True, it will download blob files, if they are present in the project, to optimize download process.
        :type download_blob_files: bool, optional
        :param skip_create_readme: Skip creating README.md file. Default is False.
        :type skip_create_readme: bool, optional
        :return: None
        :rtype: NoneType
        :Usage example:

        .. code-block:: python

                import supervisely as sly

                # Local destination Project folder
                save_directory = "/home/admin/work/supervisely/source/project"

                # Obtain server address and your api_token from environment variables
                # Edit those values if you run this notebook on your own PC
                address = os.environ['SERVER_ADDRESS']
                token = os.environ['API_TOKEN']

                # Initialize API object
                api = sly.Api(address, token)
                project_id = 8888

                # Download Project
                sly.Project.download(api, project_id, save_directory)
                project_fs = sly.Project(save_directory, sly.OpenMode.READ)
        """
        download_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            log_progress=log_progress,
            batch_size=batch_size,
            cache=cache,
            progress_cb=progress_cb,
            only_image_tags=only_image_tags,
            save_image_info=save_image_info,
            save_images=save_images,
            save_image_meta=save_image_meta,
            resume_download=resume_download,
            **kwargs,
        )

    @staticmethod
    def download_bin(
        api: sly.Api,
        project_id: int,
        dest_dir: str = None,
        dataset_ids: Optional[List[int]] = None,
        batch_size: Optional[int] = 100,
        log_progress: Optional[bool] = True,
        progress_cb: Optional[Callable] = None,
        return_bytesio: Optional[bool] = False,
    ) -> Union[str, io.BytesIO]:
        """
        Download project to the local directory in binary format. Faster than downloading project in the usual way.
        This type of project download is more suitable for creating local backups.
        It is also suitable for cases where you don't need access to individual project files, such as images or annotations.

        Binary file contains the following data:
        - ProjectInfo
        - ProjectMeta
        - List of DatasetInfo
        - List of ImageInfo
        - Dict of Figures
        - Dict of AlphaGeometries

        :param api: Supervisely API address and token.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param project_id: Project ID to download.
        :type project_id: :class:`int`
        :param dest_dir: Destination path to local directory.
        :type dest_dir: :class:`str`, optional
        :param dataset_ids: Specified list of Dataset IDs which will be downloaded. If you want to download nested datasets, you should specify all nested IDs.
        :type dataset_ids: :class:`list` [ :class:`int` ], optional
        :param batch_size: Size of a downloading batch.
        :type batch_size: :class:`int`, optional
        :param log_progress: Show downloading logs in the output.
        :type log_progress: :class:`bool`, optional
        :param progress_cb: Function for tracking download progress. Has a higher priority than log_progress.
        :type progress_cb: :class:`tqdm` or :class:`callable`, optional
        :param return_bytesio: If True, returns BytesIO object instead of saving it to the disk.
        :type return_bytesio: :class:`bool`, optional
        :return: Path to the binary file or BytesIO object.
        :rtype: :class:`str` or :class:`BytesIO`

        :Usage example:

        .. code-block:: python

                import supervisely as sly

                # Local destination Project folder
                save_directory = "/home/admin/work/supervisely/source/project"

                # Obtain server address and your api_token from environment variables
                # Edit those values if you run this notebook on your own PC
                address = os.environ['SERVER_ADDRESS']
                token = os.environ['API_TOKEN']

                # Initialize API object
                api = sly.Api(address, token)
                project_id = 8888

                # Download Project in binary format
                project_bin_path = sly.Project.download_bin(api, project_id, save_directory)
        """
        if dest_dir is None and not return_bytesio:
            raise ValueError(
                "Local save directory dest_dir must be specified if return_bytesio is False"
            )

        ds_filters = (
            [{"field": "id", "operator": "in", "value": dataset_ids}]
            if dataset_ids is not None
            else None
        )

        project_info = api.project.get_info_by_id(project_id)
        meta = ProjectMeta.from_json(api.project.get_meta(project_id, with_settings=True))

        dataset_infos = api.dataset.get_list(project_id, filters=ds_filters, recursive=True)

        image_infos = []
        figures = {}
        alpha_geometries = {}
        for dataset_info in dataset_infos:
            ds_image_infos = api.image.get_list(dataset_info.id)
            image_infos.extend(ds_image_infos)

            ds_progress = progress_cb
            if log_progress and progress_cb is None:
                ds_progress = tqdm_sly(
                    desc="Downloading dataset: {!r}".format(dataset_info.name),
                    total=len(ds_image_infos),
                )

            for batch in batched(ds_image_infos, batch_size):
                image_ids = [image_info.id for image_info in batch]
                ds_figures = api.image.figure.download(dataset_info.id, image_ids)
                alpha_ids = [
                    figure.id
                    for figures in ds_figures.values()
                    for figure in figures
                    if figure.geometry_type == sly.AlphaMask.name()
                ]
                if len(alpha_ids) > 0:
                    geometries_list = api.image.figure.download_geometries_batch(alpha_ids)
                    alpha_geometries.update(dict(zip(alpha_ids, geometries_list)))
                figures.update(ds_figures)
                if ds_progress is not None:
                    ds_progress(len(batch))
        if dataset_infos != [] and ds_progress is not None:
            ds_progress.close()
        data = (project_info, meta, dataset_infos, image_infos, figures, alpha_geometries)
        file = (
            io.BytesIO()
            if return_bytesio
            else open(os.path.join(dest_dir, f"{project_info.id}_{project_info.name}"), "wb")
        )

        if isinstance(file, io.BytesIO):
            pickle.dump(data, file)
        else:
            with file as f:
                pickle.dump(data, f)

        return file if return_bytesio else file.name

    @staticmethod
    def upload_bin(
        api: Api,
        file: Union[str, io.BytesIO],
        workspace_id: int,
        project_name: Optional[str] = None,
        with_custom_data: Optional[bool] = True,
        log_progress: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_missed: Optional[bool] = False,
    ) -> sly.ProjectInfo:
        """
        Uploads project to Supervisely from the given binary file and suitable only for projects downloaded in binary format.
        This method is a counterpart to :func:`download_bin`.
        Faster than uploading project in the usual way.

        :param api: Supervisely API address and token.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param file: Path to the binary file or BytesIO object.
        :type file: :class:`str` or :class:`BytesIO`
        :param workspace_id: Workspace ID, where project will be uploaded.
        :type workspace_id: :class:`int`
        :param project_name: Name of the project in Supervisely. Can be changed if project with the same name is already exists.
        :type project_name: :class:`str`, optional
        :param with_custom_data: If True, custom data from source project will be added to a new project.
        :type with_custom_data: :class:`bool`, optional
        :param log_progress: Show uploading progress bar.
        :type log_progress: :class:`bool`, optional
        :param progress_cb: Function for tracking upload progress for datasets. Has a higher priority than log_progress.
        :type progress_cb: tqdm or callable, optional
        :param skip_missed: Skip missed images.
        :type skip_missed: :class:`bool`, optional
        :return: ProjectInfo object.
        :rtype: :class:`ProjectInfo<supervisely.api.project.ProjectInfo>`
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            # Local folder with Project
            project_path = "/home/admin/work/supervisely/source/project/222_ProjectName"

            # Obtain server address and your api_token from environment variables
            # Edit those values if you run this notebook on your own PC
            address = os.environ['SERVER_ADDRESS']
            token = os.environ['API_TOKEN']

            # Initialize API object
            api = sly.Api(address, token)

            # Upload Project
            project_info = sly.Project.upload_bin(
                api,
                project_path,
                workspace_id=45,
                project_name="My Project"
            )
        """

        alpha_mask_name = sly.AlphaMask.name()
        project_info: sly.ProjectInfo
        meta: ProjectMeta
        dataset_infos: List[sly.DatasetInfo]
        image_infos: List[ImageInfo]
        figures: Dict[int, List[sly.FigureInfo]]  # image_id: List of figure_infos
        alpha_geometries: Dict[int, List[dict]]  # figure_id: List of geometries
        with file if isinstance(file, io.BytesIO) else open(file, "rb") as f:
            unpickler = CustomUnpickler(f)
            project_info, meta, dataset_infos, image_infos, figures, alpha_geometries = (
                unpickler.load()
            )
        if project_name is None:
            project_name = project_info.name
        new_project_info = api.project.create(
            workspace_id, project_name, change_name_if_conflict=True
        )
        custom_data = new_project_info.custom_data
        version_num = project_info.version.get("version", None) if project_info.version else 0
        custom_data["restored_from"] = {
            "project_id": project_info.id,
            "version_num": version_num + 1 if version_num is not None else "Unable to determine",
        }
        if with_custom_data:
            custom_data.update(project_info.custom_data)
        api.project.update_custom_data(new_project_info.id, custom_data, silent=True)
        new_meta = api.project.update_meta(new_project_info.id, meta)
        # remap tags
        old_tags = meta.tag_metas.to_json()
        new_tags = new_meta.tag_metas.to_json()
        old_new_tags_mapping = dict(
            map(lambda old_tag, new_tag: (old_tag["id"], new_tag["id"]), old_tags, new_tags)
        )
        # remap classes
        old_classes = meta.obj_classes.to_json()
        new_classes = new_meta.obj_classes.to_json()
        old_new_classes_mapping = dict(
            map(
                lambda old_class, new_class: (old_class["id"], new_class["id"]),
                old_classes,
                new_classes,
            )
        )
        dataset_mapping = {}
        # Sort datasets by parent, so that datasets with parent = 0 are processed first
        sorted_dataset_infos = sorted(
            dataset_infos, key=lambda dataset: (dataset.parent_id is not None, dataset.parent_id)
        )

        for dataset_info in sorted_dataset_infos:
            dataset_info: sly.DatasetInfo
            parent_ds_info = dataset_mapping.get(dataset_info.parent_id, None)
            new_parent_id = parent_ds_info.id if parent_ds_info else None
            if new_parent_id is None and dataset_info.parent_id is not None:
                logger.warning(
                    f"Parent dataset for dataset '{dataset_info.name}' not found. Will be added to project root."
                )
            new_dataset_info = api.dataset.create(
                new_project_info.id, dataset_info.name, parent_id=new_parent_id
            )
            if new_dataset_info is None:
                raise RuntimeError(f"Failed to restore dataset {dataset_info.name}")
            dataset_mapping[dataset_info.id] = new_dataset_info
        info_values_by_dataset = defaultdict(
            lambda: {"infos": [], "ids": [], "names": [], "hashes": [], "metas": [], "links": []}
        )

        if skip_missed:
            existing_hashes = api.image.check_existing_hashes(
                list(set([inf.hash for inf in image_infos if inf.hash and not inf.link]))
            )
            workspace_info = api.workspace.get_info_by_id(workspace_id)
            existing_links = api.image.check_existing_links(
                list(set([inf.link for inf in image_infos if inf.link])),
                team_id=workspace_info.team_id,
            )
        image_infos = sorted(image_infos, key=lambda info: info.link is not None)

        values_lists = ["infos", "ids", "names", "hashes", "metas", "links"]
        attributes = [None, "id", "name", "hash", "meta", "link"]
        for info in image_infos:
            # pylint: disable=possibly-used-before-assignment
            if skip_missed and info.hash and not info.link:
                if info.hash not in existing_hashes:
                    logger.warning(
                        f"Image with name {info.name} can't be uploaded. Hash {info.hash} not found"
                    )
                    continue
            if skip_missed and info.link:
                if info.link not in existing_links:
                    logger.warning(
                        f"Image with name {info.name} can't be uploaded. Link {info.link} can't be accessed"
                    )
                    continue
            for value_list, attr in zip(values_lists, attributes):
                if value_list == "infos":
                    info_values_by_dataset[info.dataset_id][value_list].append(info)
                else:
                    info_values_by_dataset[info.dataset_id][value_list].append(getattr(info, attr))

        for dataset_id, values in info_values_by_dataset.items():
            dataset_name = None
            if dataset_id in dataset_mapping:
                # return new dataset_id and name
                new_ds_info = dataset_mapping.get(dataset_id)
                dataset_id, dataset_name = new_ds_info.id, new_ds_info.name
                if dataset_id is None:
                    raise KeyError(f"Dataset ID {dataset_id} not found in mapping")

            ds_progress = progress_cb
            if log_progress and progress_cb is None:
                ds_progress = tqdm_sly(
                    desc="Uploading images to {!r}".format(dataset_name),
                    total=len(values["names"]),
                )

            # ------------------------------------ Determine Upload Method ----------------------------------- #

            none_link_indices = [i for i, link in enumerate(values["links"]) if link is None]

            if len(none_link_indices) == len(values["links"]):
                new_file_infos = api.image.upload_hashes(
                    dataset_id,
                    names=values["names"],
                    hashes=values["hashes"],
                    metas=values["metas"],
                    batch_size=200,
                    progress_cb=ds_progress,
                )
            elif not none_link_indices:
                new_file_infos = api.image.upload_links(
                    dataset_id,
                    names=values["names"],
                    links=values["links"],
                    metas=values["metas"],
                    batch_size=200,
                    progress_cb=ds_progress,
                )
            else:
                if not all(
                    none_link_indices[i] - none_link_indices[i - 1] == 1
                    for i in range(1, len(none_link_indices))
                ):
                    raise ValueError(
                        "Internal upload_bin Error. Images with links and without links are not in continuous blocks"
                    )
                i = none_link_indices[0]  # first image without link
                j = none_link_indices[-1]  # last image without link

                new_file_infos = api.image.upload_hashes(
                    dataset_id,
                    names=values["names"][i : j + 1],
                    hashes=values["hashes"][i : j + 1],
                    metas=values["metas"][i : j + 1],
                    batch_size=200,
                    progress_cb=ds_progress,
                )
                new_file_infos_link = api.image.upload_links(
                    dataset_id,
                    names=values["names"][j + 1 :],
                    links=values["links"][j + 1 :],
                    metas=values["metas"][j + 1 :],
                    batch_size=200,
                    progress_cb=ds_progress,
                )
                new_file_infos.extend(new_file_infos_link)
            # ----------------------------------------------- - ---------------------------------------------- #

            # image_lists_by_tags -> tagId: {tagValue: [imageId]}
            image_lists_by_tags = defaultdict(lambda: defaultdict(list))
            alpha_figures = []
            other_figures = []
            all_figure_tags = defaultdict(list)  # figure_id: List of (tagId, value)
            old_alpha_figure_ids = []
            tags_list = []  # to append tags to figures in bulk
            if ds_progress is not None:
                ds_fig_progress = tqdm_sly(
                    desc="Processing figures for images in {!r}".format(dataset_name),
                    total=len(new_file_infos),
                )
            for old_file_info, new_file_info in zip(values["infos"], new_file_infos):
                for tag in old_file_info.tags:
                    new_tag_id = old_new_tags_mapping[tag.get("tagId")]
                    image_lists_by_tags[new_tag_id][tag.get("value")].append(new_file_info.id)
                image_figures = figures.get(old_file_info.id, [])
                if len(image_figures) > 0:
                    alpha_figure_jsons = []
                    other_figure_jsons = []
                    for figure in image_figures:
                        figure_json = figure._asdict()
                        if figure.geometry_type == alpha_mask_name:
                            alpha_figure_jsons.append(figure_json)
                            old_alpha_figure_ids.append(figure_json["id"])
                        else:
                            other_figure_jsons.append(figure_json)

                    def create_figure_json(figure, geometry):
                        return {
                            "meta": figure["meta"] if figure["meta"] is not None else {},
                            "entityId": new_file_info.id,
                            "classId": old_new_classes_mapping[figure["class_id"]],
                            "geometry": geometry,
                            "geometryType": figure["geometry_type"],
                        }

                    new_figure_jsons = [
                        create_figure_json(figure, figure["geometry"])
                        for figure in other_figure_jsons
                    ]
                    new_alpha_figure_jsons = [
                        create_figure_json(figure, None) for figure in alpha_figure_jsons
                    ]
                    other_figures.extend(new_figure_jsons)
                    alpha_figures.extend(new_alpha_figure_jsons)

                    def process_figures(figure_jsons, figure_tags):
                        for figure in figure_jsons:
                            figure_tags[figure.get("id")].extend(
                                (tag.get("tagId"), tag.get("value", None)) for tag in figure["tags"]
                            )

                    process_figures(other_figure_jsons, all_figure_tags)
                    process_figures(alpha_figure_jsons, all_figure_tags)
                if ds_progress is not None:
                    ds_fig_progress.update(1)
            all_figure_ids = api.image.figure.create_bulk(
                other_figures,
                dataset_id=new_file_info.dataset_id,
            )
            new_alpha_figure_ids = api.image.figure.create_bulk(
                alpha_figures, dataset_id=new_file_info.dataset_id
            )
            all_figure_ids.extend(new_alpha_figure_ids)
            ordered_alpha_geometries = list(map(alpha_geometries.get, old_alpha_figure_ids))
            api.image.figure.upload_geometries_batch(new_alpha_figure_ids, ordered_alpha_geometries)
            for tag, value in image_lists_by_tags.items():
                for value, image_ids in value.items():
                    api.image.add_tag_batch(image_ids, tag, value, batch_size=200)
            for new_of_id, tags in zip(all_figure_ids, all_figure_tags.values()):
                for tag_id, tag_value in tags:
                    new_tag_id = old_new_tags_mapping[tag_id]
                    tags_list.append(
                        {"tagId": new_tag_id, "figureId": new_of_id, "value": tag_value}
                    )

            api.image.tag.add_to_objects(
                new_project_info.id,
                tags_list,
                batch_size=300,
                log_progress=True if ds_progress is not None else False,
            )
        return new_project_info

    @staticmethod
    def upload(
        dir: str,
        api: Api,
        workspace_id: int,
        project_name: Optional[str] = None,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> Tuple[int, str]:
        """
        Uploads project to Supervisely from the given directory.

        If you have a metadata.json files in the project directory for images, you will be able to upload images with added custom sort parameter.
        To do this, use context manager :func:`api.image.add_custom_sort` with the desired key name from the metadata.json file which will be used for sorting.
        More about project struture: https://developer.supervisely.com/getting-started/supervisely-annotation-format/project-structure#project-structure-example
        Refer to the example section for usage details.

        :param dir: Path to project directory.
        :type dir: :class:`str`
        :param api: Supervisely API address and token.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param workspace_id: Workspace ID, where project will be uploaded.
        :type workspace_id: :class:`int`
        :param project_name: Name of the project in Supervisely. Can be changed if project with the same name is already exists.
        :type project_name: :class:`str`, optional
        :param log_progress: Show uploading progress bar.
        :type log_progress: :class:`bool`
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: Project ID and name. It is recommended to check that returned project name coincides with provided project name.
        :rtype: :class:`int`, :class:`str`
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            # Local folder with Project
            project_directory = "/home/admin/work/supervisely/source/project"

            # Obtain server address and your api_token from environment variables
            # Edit those values if you run this notebook on your own PC
            address = os.environ['SERVER_ADDRESS']
            token = os.environ['API_TOKEN']

            # Initialize API object
            api = sly.Api(address, token)

            # Upload Project
            project_id, project_name = sly.Project.upload(
                project_directory,
                api,
                workspace_id=45,
                project_name="My Project"
            )

            # Upload project with added custom sort order
            # This context manager processes every image and adds a custom sort order
            # if `meta` is present in the image info file or image meta file.
            # Otherwise, it will be uploaded without a custom sort order.
            with api.image.add_custom_sort(key="key_name"):
                project_id, project_name = sly.Project.upload(
                    project_directory,
                    api,
                    workspace_id=45,
                    project_name="My Project"
                )
        """
        return upload_project(
            dir=dir,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    @staticmethod
    async def download_async(
        api: Api,
        project_id: int,
        dest_dir: str,
        dataset_ids: Optional[List[int]] = None,
        log_progress: bool = True,
        semaphore: asyncio.Semaphore = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        only_image_tags: Optional[bool] = False,
        save_image_info: Optional[bool] = False,
        save_images: bool = True,
        save_image_meta: bool = False,
        images_ids: Optional[List[int]] = None,
        resume_download: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        Download project from Supervisely to the given directory in asynchronous mode.

        :param api: Supervisely API address and token.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param project_id: Supervisely downloadable project ID.
        :type project_id: :class:`int`
        :param dest_dir: Destination directory.
        :type dest_dir: :class:`str`
        :param dataset_ids: Filter datasets by IDs.
        :type dataset_ids: :class:`list` [ :class:`int` ], optional
        :param log_progress: Show uploading progress bar.
        :type log_progress: :class:`bool`
        :param semaphore: Semaphore to limit the number of concurrent downloads of items.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param only_image_tags: Download project with only images tags (without objects tags).
        :type only_image_tags: :class:`bool`, optional
        :param save_image_info: Download images infos or not.
        :type save_image_info: :class:`bool`, optional
        :param save_images: Download images or not.
        :type save_images: :class:`bool`, optional
        :param save_image_meta: Download images metadata in JSON format or not.
        :type save_image_meta: :class:`bool`, optional
        :param images_ids: Filter images by IDs.
        :type images_ids: :class:`list` [ :class:`int` ], optional
        :param resume_download: Resume download enables to download only missing files avoiding erase of existing files.
        :type resume_download: :class:`bool`, optional
        :param skip_create_readme: Skip creating README.md file. Default is False.
        :type skip_create_readme: bool, optional
        :return: None
        :rtype: NoneType

        :Usage example:

            .. code-block:: python

                import supervisely as sly
                from supervisely._utils import run_coroutine

                os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
                os.environ['API_TOKEN'] = 'Your Supervisely API Token'
                api = sly.Api.from_env()

                project_id = 8888
                save_directory = "/path/to/save/projects"

                coroutine = sly.Project.download_async(api, project_id, save_directory)
                run_coroutine(coroutine)
        """
        if kwargs.pop("cache", None) is not None:
            logger.warning(
                "Cache is not supported in async mode and will be ignored. "
                "Use resume_download parameter instead to optimize download process."
            )

        await _download_project_async(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            log_progress=log_progress,
            semaphore=semaphore,
            only_image_tags=only_image_tags,
            save_image_info=save_image_info,
            save_images=save_images,
            progress_cb=progress_cb,
            save_image_meta=save_image_meta,
            images_ids=images_ids,
            resume_download=resume_download,
            **kwargs,
        )

    def to_coco(
        self,
        dest_dir: Optional[str] = None,
        copy_images: bool = False,
        with_captions: bool = False,
        log_progress: bool = True,
        progress_cb: Optional[Callable] = None,
    ) -> None:
        """
        Convert Supervisely project to COCO format.

        :param dest_dir: Destination directory.
        :type dest_dir: :class:`str`, optional
        :param copy_images: Copy images to the destination directory.
        :type copy_images: :class:`bool`
        :param with_captions: Return captions for images.
        :type with_captions: :class:`bool`
        :param log_progress: Show uploading progress bar.
        :type log_progress: :class:`bool`
        :param progress_cb: Function for tracking conversion progress (for all items in the project).
        :type progress_cb: callable, optional
        :return: None
        :rtype: NoneType

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            # Local folder with Project
            project_directory = "/home/admin/work/supervisely/source/project"

            # Convert Project to COCO format
            sly.Project(project_directory).to_coco(log_progress=True)
            # or
            from supervisely.convert import to_coco
            to_coco(project_directory, dest_dir="./coco_project")
        """
        from supervisely.convert import project_to_coco

        project_to_coco(
            project=self,
            dest_dir=dest_dir,
            copy_images=copy_images,
            with_captions=with_captions,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    def to_yolo(
        self,
        dest_dir: Optional[str] = None,
        task_type: Literal["detect", "segment", "pose"] = "detect",
        log_progress: bool = True,
        progress_cb: Optional[Callable] = None,
        val_datasets: Optional[List[str]] = None,
    ) -> None:
        """
        Convert Supervisely project to YOLO format.

        :param dest_dir: Destination directory.
        :type dest_dir: :class:`str`, optional
        :param task_type: Task type for YOLO format. Possible values: 'detection', 'segmentation', 'pose'.
        :type task_type: :class:`str` or :class:`TaskType`, optional
        :param log_progress: Show uploading progress bar.
        :type log_progress: :class:`bool`
        :param progress_cb: Function for tracking conversion progress (for all items in the project).
        :type progress_cb: callable, optional
        :param val_datasets:    List of dataset names for validation.
                            Full dataset names are required (e.g., 'ds0/nested_ds1/ds3').
                            If specified, datasets from the list will be marked as val, others as train.
                            If not specified, the function will determine the validation datasets automatically.
        :type val_datasets: :class:`list` [ :class:`str` ], optional
        :return: None
        :rtype: NoneType

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            # Local folder with Project
            project_directory = "/home/admin/work/supervisely/source/project"

            # Convert Project to YOLO format
            sly.Project(project_directory).to_yolo(log_progress=True)
            # or
            from supervisely.convert import to_yolo
            to_yolo(project_directory, dest_dir="./yolo_project")
        """

        from supervisely.convert import project_to_yolo

        return project_to_yolo(
            project=self,
            dest_dir=dest_dir,
            task_type=task_type,
            log_progress=log_progress,
            progress_cb=progress_cb,
            val_datasets=val_datasets,
        )

    def to_pascal_voc(
        self,
        dest_dir: Optional[str] = None,
        train_val_split_coef: float = 0.8,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Convert Supervisely project to Pascal VOC format.

        :param dest_dir: Destination directory.
        :type dest_dir: :class:`str`, optional
        :param train_val_split_coef: Coefficient for splitting images into train and validation sets.
        :type train_val_split_coef: :class:`float`, optional
        :param log_progress: Show uploading progress bar.
        :type log_progress: :class:`bool`
        :param progress_cb: Function for tracking conversion progress (for all items in the project).
        :type progress_cb: callable, optional
        :return: None
        :rtype: NoneType

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            # Local folder with Project
            project_directory = "/home/admin/work/supervisely/source/project"

            # Convert Project to YOLO format
            sly.Project(project_directory).to_pascal_voc(log_progress=True)
            # or
            from supervisely.convert import to_pascal_voc
            to_pascal_voc(project_directory, dest_dir="./pascal_voc_project")
        """
        from supervisely.convert import project_to_pascal_voc

        project_to_pascal_voc(
            project=self,
            dest_dir=dest_dir,
            train_val_split_coef=train_val_split_coef,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )


def read_single_project(
    dir: str,
    project_class: Optional[
        Union[
            Project,
            sly.VideoProject,
            sly.VolumeProject,
            sly.PointcloudProject,
            sly.PointcloudEpisodeProject,
        ]
    ] = Project,
) -> Union[
    Project,
    sly.VideoProject,
    sly.VolumeProject,
    sly.PointcloudProject,
    sly.PointcloudEpisodeProject,
]:
    """
    Read project from given directory or tries to find project directory in subdirectories.

    :param dir: Path to directory, which contains project folder or have project folder in any subdirectory.
    :type dir: :class:`str`
    :param project_class: Project object of arbitrary modality
    :type project_class: :class: `Project` or `VideoProject` or `VolumeProject` or `PointcloudProject` or `PointcloudEpisodeProject`, optional

    :return: Project class object of arbitrary modality
    :rtype: :class: `Project` or `VideoProject` or `VolumeProject` or `PointcloudProject` or `PointcloudEpisodeProject`
    :raises: RuntimeError if the given directory and it's subdirectories contains more than one valid project folder.
    :raises: FileNotFoundError if the given directory or any of it's subdirectories doesn't contain valid project folder.

    :Usage example:
     .. code-block:: python
        import supervisely as sly
        proj_dir = "/home/admin/work/supervisely/source/project" # Project directory or directory with project subdirectory.
        project = sly.read_single_project(proj_dir)
    """
    project_dirs = [project_dir for project_dir in find_project_dirs(dir, project_class)]
    if len(project_dirs) > 1:
        raise RuntimeError(
            f"The given directory {dir} and it's subdirectories contains more than one valid project folder. "
            f"The following project folders were found: {project_dirs}. "
            "Ensure that you have only one project in the given directory and it's subdirectories."
        )
    elif len(project_dirs) == 0:
        raise FileNotFoundError(
            f"The given directory {dir} or any of it's subdirectories doesn't contain valid project folder."
        )
    return project_class(project_dirs[0], OpenMode.READ)


def find_project_dirs(dir: str, project_class: Optional[Project] = Project) -> Generator[str]:
    """Yields directories, that contain valid project folder in the given directory or in any of it's subdirectories.
    :param dir: Path to directory, which contains project folder or have project folder in any subdirectory.
    :type dir: str
    :param project_class: Project object
    :type project_class: :class:`Project<Project>`
    :return: Path to directory, that contain meta.json file.
    :rtype: str
    :Usage example:
     .. code-block:: python
        import supervisely as sly
        # Local folder (or any of it's subdirectories) which contains sly.Project files.
        input_directory = "/home/admin/work/supervisely/source"
        for project_dir in sly.find_project_dirs(input_directory):
            project_fs = sly.Project(meta_json_dir, sly.OpenMode.READ)
            # Do something with project_fs
    """
    paths = list_dir_recursively(dir)
    for path in paths:
        if get_file_name_with_ext(path) == "meta.json":
            parent_dir = os.path.dirname(path)
            project_dir = os.path.join(dir, parent_dir)
            try:
                project_class(project_dir, OpenMode.READ)
                yield project_dir
            except Exception:
                pass


def _download_project(
    api: sly.Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: bool = True,
    batch_size: Optional[int] = 50,
    only_image_tags: Optional[bool] = False,
    save_image_info: Optional[bool] = False,
    save_images: Optional[bool] = True,
    progress_cb: Optional[Callable] = None,
    save_image_meta: Optional[bool] = False,
    images_ids: Optional[List[int]] = None,
    resume_download: Optional[bool] = False,
    **kwargs,
):
    download_blob_files = kwargs.pop("download_blob_files", False)
    skip_create_readme = kwargs.pop("skip_create_readme", False)

    dataset_ids = set(dataset_ids) if (dataset_ids is not None) else None
    project_fs = None

    meta = ProjectMeta.from_json(api.project.get_meta(project_id, with_settings=True))
    if os.path.exists(dest_dir) and resume_download:
        dump_json_file(meta.to_json(), os.path.join(dest_dir, "meta.json"))
        try:
            project_fs = Project(dest_dir, OpenMode.READ)
        except RuntimeError as e:
            if "Project is empty" in str(e):
                clean_dir(dest_dir)
                project_fs = None
            else:
                raise
    if project_fs is None:
        project_fs = Project(dest_dir, OpenMode.CREATE)
    project_fs.set_meta(meta)

    if progress_cb is not None:
        log_progress = False

    id_to_tagmeta = None
    if only_image_tags is True:
        id_to_tagmeta = meta.tag_metas.get_id_mapping()

    existing_datasets = {dataset.path: dataset for dataset in project_fs.datasets}
    for parents, dataset in api.dataset.tree(project_id):
        blob_files_to_download = {}
        dataset_path = Dataset._get_dataset_path(dataset.name, parents)
        dataset_id = dataset.id
        if dataset_ids is not None and dataset_id not in dataset_ids:
            continue

        if dataset_path in existing_datasets:
            dataset_fs = existing_datasets[dataset_path]
        else:
            dataset_fs = project_fs.create_dataset(dataset.name, dataset_path)

        all_images = api.image.get_list(dataset_id, force_metadata_for_links=False)
        images = [image for image in all_images if images_ids is None or image.id in images_ids]
        ds_total = len(images)

        ds_progress = progress_cb
        if log_progress is True:
            ds_progress = tqdm_sly(
                desc="Downloading images from {!r}".format(dataset.name),
                total=ds_total,
            )

        anns_progress = None
        if log_progress or progress_cb is not None:
            anns_progress = tqdm_sly(
                desc="Downloading annotations from {!r}".format(dataset.name),
                total=ds_total,
                leave=False,
            )

        with ApiContext(
            api,
            project_id=project_id,
            dataset_id=dataset_id,
            project_meta=meta,
        ):
            for batch in batched(images, batch_size):
                batch: List[ImageInfo]
                image_ids = [image_info.id for image_info in batch]
                image_names = [image_info.name for image_info in batch]

                existing_image_infos: Dict[str, ImageInfo] = {}
                for image_name in image_names:
                    try:
                        image_info = dataset_fs.get_item_info(image_name)
                    except:
                        image_info = None
                    existing_image_infos[image_name] = image_info

                indexes_to_download = []
                for i, image_info in enumerate(batch):
                    existing_image_info = existing_image_infos[image_info.name]
                    if (
                        existing_image_info is None
                        or existing_image_info.updated_at != image_info.updated_at
                    ):
                        indexes_to_download.append(i)

                # Collect images that was added to the project as offsets from archive in Team Files
                indexes_with_offsets = []
                for idx in indexes_to_download:
                    image_info: ImageInfo = batch[idx]
                    if image_info.related_data_id is not None:
                        blob_files_to_download[image_info.related_data_id] = image_info.download_id
                        indexes_with_offsets.append(idx)

                # Download images in numpy format
                batch_imgs_bytes = [None] * len(image_ids)
                if save_images and indexes_to_download:

                    # For a lot of small files that stored in blob file. Downloads blob files to optimize download process.
                    if download_blob_files and len(indexes_with_offsets) > 0:
                        bytes_indexes_to_download = indexes_to_download.copy()
                        for blob_file_id, download_id in blob_files_to_download.items():
                            if blob_file_id not in project_fs.blob_files:
                                api.image.download_blob_file(
                                    project_id=project_id,
                                    download_id=download_id,
                                    path=os.path.join(project_fs.blob_dir, f"{blob_file_id}.tar"),
                                    log_progress=(
                                        True if log_progress or progress_cb is not None else False
                                    ),
                                )
                                project_fs.add_blob_file(blob_file_id)

                            # Process blob image offsets
                            offsets_file_name = f"{blob_file_id}{OFFSETS_PKL_SUFFIX}"
                            offsets_file_path = os.path.join(
                                dataset_fs.directory, offsets_file_name
                            )

                            # Initialize counter for total image offsets for this blob file
                            total_offsets_count = 0
                            current_batch = []

                            # Get offsets from image infos
                            for idx in indexes_with_offsets:
                                image_info = batch[idx]
                                if image_info.related_data_id == blob_file_id:
                                    blob_image_info = BlobImageInfo(
                                        name=image_info.name,
                                        offset_start=image_info.offset_start,
                                        offset_end=image_info.offset_end,
                                    )
                                    current_batch.append(blob_image_info)
                                    bytes_indexes_to_download.remove(idx)

                                    # When batch size is reached, dump to file
                                    if len(current_batch) >= OFFSETS_PKL_BATCH_SIZE:
                                        BlobImageInfo.dump_to_pickle(
                                            current_batch, offsets_file_path
                                        )
                                        total_offsets_count += len(current_batch)
                                        current_batch = []
                            # Dump any remaining items in the last batch
                            if len(current_batch) > 0:
                                BlobImageInfo.dump_to_pickle(current_batch, offsets_file_path)
                                total_offsets_count += len(current_batch)

                            if total_offsets_count > 0:
                                logger.debug(
                                    f"Saved {total_offsets_count} image offsets for {blob_file_id} to {offsets_file_path} in {(total_offsets_count + OFFSETS_PKL_BATCH_SIZE - 1) // OFFSETS_PKL_BATCH_SIZE} batches"
                                )
                                ds_progress(total_offsets_count)

                            image_ids_to_download = [
                                image_ids[i] for i in bytes_indexes_to_download
                            ]
                            for index, img in zip(
                                bytes_indexes_to_download,
                                api.image.download_bytes(
                                    dataset_id,
                                    image_ids_to_download,
                                    progress_cb=ds_progress,
                                ),
                            ):
                                batch_imgs_bytes[index] = img
                    # If you want to download images in classic way
                    else:
                        image_ids_to_download = [image_ids[i] for i in indexes_to_download]
                        for index, img in zip(
                            indexes_to_download,
                            api.image.download_bytes(
                                dataset_id,
                                image_ids_to_download,
                                progress_cb=ds_progress,
                            ),
                        ):
                            batch_imgs_bytes[index] = img

                if ds_progress is not None:
                    ds_progress(len(batch) - len(indexes_to_download))

                # download annotations in json format
                ann_jsons = [None] * len(image_ids)
                if only_image_tags is False:
                    if indexes_to_download:
                        for index, ann_info in zip(
                            indexes_to_download,
                            api.annotation.download_batch(
                                dataset_id,
                                [image_ids[i] for i in indexes_to_download],
                                progress_cb=anns_progress,
                            ),
                        ):
                            ann_jsons[index] = ann_info.annotation
                else:
                    if indexes_to_download:
                        for index in indexes_to_download:
                            image_info = batch[index]
                            tags = TagCollection.from_api_response(
                                image_info.tags,
                                meta.tag_metas,
                                id_to_tagmeta,
                            )
                            tmp_ann = Annotation(
                                img_size=(image_info.height, image_info.width), img_tags=tags
                            )
                            ann_jsons[index] = tmp_ann.to_json()
                            if anns_progress is not None:
                                anns_progress(len(indexes_to_download))
                if anns_progress is not None:
                    anns_progress(len(batch) - len(indexes_to_download))

                for img_info, name, img_bytes, ann in zip(
                    batch, image_names, batch_imgs_bytes, ann_jsons
                ):
                    dataset_fs: Dataset
                    # to fix already downloaded images that doesn't have info files
                    dataset_fs.delete_item(name)
                    dataset_fs.add_item_raw_bytes(
                        item_name=name,
                        item_raw_bytes=img_bytes if save_images is True else None,
                        ann=dataset_fs.get_ann(name, meta) if ann is None else ann,
                        img_info=img_info if save_image_info is True else None,
                    )

        if save_image_meta:
            meta_dir = dataset_fs.meta_dir
            for image_info in images:
                if image_info.meta:
                    sly.fs.mkdir(meta_dir)
                    sly.json.dump_json_file(
                        image_info.meta, dataset_fs.get_item_meta_path(image_info.name)
                    )

        # delete redundant items
        items_names_set = set([img.name for img in all_images])
        for item_name in dataset_fs.get_items_names():
            if item_name not in items_names_set:
                dataset_fs.delete_item(item_name)
    if not skip_create_readme:
        try:
            if download_blob_files:
                project_info = api.project.get_info_by_id(project_id)
                create_blob_readme(project_fs=project_fs, project_info=project_info, api=api)
            else:
                create_readme(dest_dir, project_id, api)
        except Exception as e:
            logger.info(f"There was an error while creating README: {e}")


def upload_project(
    dir: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    project_id: Optional[int] = None,
) -> Tuple[int, str]:
    project_fs = read_single_project(dir)

    if not project_id:
        if project_name is None:
            project_name = project_fs.name

        if api.project.exists(workspace_id, project_name):
            project_name = api.project.get_free_name(workspace_id, project_name)

        project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    else:
        project = api.project.get_info_by_id(project_id)
    updated_meta = api.project.update_meta(project.id, project_fs.meta.to_json())

    if progress_cb is not None:
        log_progress = False

    # image_id_dct, anns_paths_dct = {}, {}
    dataset_map = {}

    total_blob_size = 0
    upload_blob_progress = None
    src_paths = []
    dst_paths = []
    for blob_file in project_fs.blob_files:
        if log_progress:
            total_blob_size += os.path.getsize(os.path.join(project_fs.blob_dir, blob_file))
        src_paths.append(os.path.join(project_fs.blob_dir, blob_file))
        dst_paths.append(os.path.join(f"/{TF_BLOB_DIR}", blob_file))
    if log_progress and len(src_paths) > 0:
        upload_blob_progress = tqdm_sly(
            desc="Uploading blob files", total=total_blob_size, unit="B", unit_scale=True
        )
    if len(src_paths) > 0:
        blob_file_infos = api.file.upload_bulk(
            team_id=project.team_id,
            src_paths=src_paths,
            dst_paths=dst_paths,
            progress_cb=upload_blob_progress,
        )
    else:
        blob_file_infos = []

    for ds_fs in project_fs.datasets:
        logger.debug(f"Processing dataset: {ds_fs.name}")
        if len(ds_fs.parents) > 0:
            parent = f"{os.path.sep}".join(ds_fs.parents)
            parent_id = dataset_map.get(parent)

        else:
            parent = ""
            parent_id = None
        dataset = api.dataset.create(project.id, ds_fs.short_name, parent_id=parent_id)
        dataset_map[os.path.join(parent, dataset.name)] = dataset.id
        ds_fs: Dataset

        with ApiContext(
            api,
            project_id=project.id,
            dataset_id=dataset.id,
            project_meta=updated_meta,
        ):
            names, img_paths, img_infos, ann_paths = [], [], [], []
            for item_name in ds_fs:
                img_path, ann_path = ds_fs.get_item_paths(item_name)
                img_info_path = ds_fs.get_img_info_path(item_name)

                names.append(item_name)
                img_paths.append(img_path)
                ann_paths.append(ann_path)

                if os.path.isfile(img_info_path):
                    img_infos.append(ds_fs.get_image_info(item_name=item_name))
                else:
                    img_infos.append(None)

            # img_paths = list(filter(lambda x: os.path.isfile(x), img_paths))
            source_img_paths_len = len(img_paths)
            valid_indices = []
            valid_paths = []
            offset_indices = []
            for i, path in enumerate(img_paths):
                if os.path.isfile(path):
                    valid_indices.append(i)
                    valid_paths.append(path)
                elif len(project_fs.blob_files) > 0:
                    offset_indices.append(i)
                else:
                    if img_infos[i] is not None:
                        logger.debug(f"Image will be uploaded by image_info: {names[i]}")
                    else:
                        logger.warning(
                            f"Image and image info file not found, image will be skipped: {names[i]}"
                        )
            img_paths = valid_paths
            ann_paths = list(filter(lambda x: os.path.isfile(x), ann_paths))
            # Create a mapping from name to index position for quick lookups
            offset_name_to_idx = {names[i]: i for i in offset_indices}
            metas = [{} for _ in names]

            img_infos_count = sum(1 for item in img_infos if item is not None)

            if len(img_paths) == 0 and img_infos_count == 0 and len(offset_indices) == 0:
                # Dataset is empty
                continue

            meta_dir = os.path.join(dir, ds_fs.name, "meta")
            if os.path.isdir(meta_dir):
                metas = []
                for name in names:
                    meta_path = os.path.join(meta_dir, name + ".json")
                    if os.path.isfile(meta_path):
                        metas.append(sly.json.load_json_file(meta_path))
                    else:
                        metas.append({})

            ds_progress = progress_cb
            if log_progress is True:
                ds_progress = tqdm_sly(
                    desc="Uploading images to {!r}".format(dataset.name),
                    total=len(names),
                )

            if img_infos_count != 0:
                merged_metas = []
                for img_info, meta in zip(img_infos, metas):
                    if img_info is None:
                        merged_metas.append(meta)
                        continue
                    merged_meta = {**(img_info.meta or {}), **meta}
                    merged_metas.append(merged_meta)
                metas = merged_metas

            if len(img_paths) != 0 or len(offset_indices) != 0:

                uploaded_img_infos = [None] * source_img_paths_len
                uploaded_img_infos_paths = api.image.upload_paths(
                    dataset_id=dataset.id,
                    names=[name for i, name in enumerate(names) if i in valid_indices],
                    paths=img_paths,
                    progress_cb=ds_progress,
                    metas=[metas[i] for i in valid_indices],
                )
                for i, img_info in zip(valid_indices, uploaded_img_infos_paths):
                    uploaded_img_infos[i] = img_info
                for blob_offsets in ds_fs.blob_offsets:
                    blob_file = None
                    for blob_file_info in blob_file_infos:
                        if Path(blob_file_info.name).stem == removesuffix(
                            Path(blob_offsets).name, OFFSETS_PKL_SUFFIX
                        ):
                            blob_file = blob_file_info
                            break

                    if blob_file is None:
                        raise ValueError(
                            f"Cannot find blob file for offsets: {blob_offsets}. "
                            f"Check the Team File directory '{TF_BLOB_DIR}', corresponding blob file should be uploaded."
                        )
                    uploaded_img_infos_offsets = api.image.upload_by_offsets_generator(
                        dataset=dataset,
                        team_file_id=blob_file.id,
                        offsets_file_path=blob_offsets,
                        progress_cb=ds_progress,
                        metas={names[i]: metas[i] for i in offset_indices},
                    )
                    for img_info_batch in uploaded_img_infos_offsets:
                        for img_info in img_info_batch:
                            idx = offset_name_to_idx.get(img_info.name)
                            if idx is not None:
                                uploaded_img_infos[idx] = img_info
            elif img_infos_count != 0:
                if img_infos_count != len(names):
                    raise ValueError(
                        f"Cannot upload Project: image info files count ({img_infos_count}) doesn't match with images count ({len(names)}) that are going to be uploaded. "
                        "Check the directory structure, all annotation files should have corresponding image info files."
                    )
                uploaded_img_infos = api.image.upload_ids(
                    dataset_id=dataset.id,
                    names=names,
                    ids=[img_info.id for img_info in img_infos],
                    progress_cb=ds_progress,
                    metas=metas,
                )
            else:
                raise ValueError(
                    "Cannot upload Project: img_paths is empty and img_infos_paths is empty"
                )
            # image_id_dct[ds_fs.name] =
            image_ids = [img_info.id for img_info in uploaded_img_infos]
            # anns_paths_dct[ds_fs.name] = ann_paths

            anns_progress = None
            if log_progress or progress_cb is not None:
                anns_progress = tqdm_sly(
                    desc="Uploading annotations to {!r}".format(dataset.name),
                    total=len(image_ids),
                    leave=False,
                )
            api.annotation.upload_paths(image_ids, ann_paths, anns_progress)

    return project.id, project.name


def download_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: bool = True,
    batch_size: Optional[int] = 50,
    cache: Optional[FileCache] = None,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    only_image_tags: Optional[bool] = False,
    save_image_info: Optional[bool] = False,
    save_images: bool = True,
    save_image_meta: bool = False,
    images_ids: Optional[List[int]] = None,
    resume_download: Optional[bool] = False,
    **kwargs,
) -> None:
    """
    Download image project to the local directory.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Project ID to download
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded.
    :type dataset_ids: list(int), optional
    :param log_progress: Show downloading logs in the output. By default, it is True.
    :type log_progress: bool, optional
    :param batch_size: Size of a downloading batch.
    :type batch_size: int, optional
    :param cache: Cache of downloading files.
    :type cache: FileCache, optional
    :param progress_cb: Function for tracking download progress.
    :type progress_cb: tqdm or callable, optional
    :param only_image_tags: Specify if downloading images only with image tags. Alternatively, full annotations will be downloaded.
    :type only_image_tags: bool, optional
    :param save_image_info: Include image info in the download.
    :type save_image_info, bool, optional
    :param save_images: Include images in the download.
    :type save_images, bool, optional
    :param save_image_meta: Include images metadata in JSON format in the download.
    :type save_imgge_meta: bool, optional
    :param images_ids: Specified list of Image IDs which will be downloaded.
    :type images_ids: list(int), optional
    :param resume_download: Resume download enables to download only missing files avoiding erase of existing files.
    :type resume_download: bool, optional
    :param download_blob_files: Default is False. It will download images in classic way.
                                If True, it will download blob files, if they are present in the project, to optimize download process.
    :type download_blob_files: bool, optional
    :param skip_create_readme: Skip creating README.md file. Default is False.
    :type skip_create_readme: bool, optional
    :return: None.
    :rtype: NoneType
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        from tqdm import tqdm
        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

        dest_dir = 'your/local/dest/dir'

        # Download image project
        project_id = 17732
        project_info = api.project.get_info_by_id(project_id)
        num_images = project_info.items_count

        p = tqdm(desc="Downloading image project", total=num_images)
        sly.download(
            api,
            project_id,
            dest_dir,
            progress_cb=p,
        )
    """
    if cache is None:
        _download_project(
            api,
            project_id,
            dest_dir,
            dataset_ids,
            log_progress,
            batch_size,
            only_image_tags=only_image_tags,
            save_image_info=save_image_info,
            save_images=save_images,
            progress_cb=progress_cb,
            save_image_meta=save_image_meta,
            images_ids=images_ids,
            resume_download=resume_download,
            **kwargs,
        )
    else:
        _download_project_optimized(
            api,
            project_id,
            dest_dir,
            dataset_ids,
            cache,
            progress_cb,
            only_image_tags=only_image_tags,
            save_image_info=save_image_info,
            save_images=save_images,
            log_progress=log_progress,
            images_ids=images_ids,
            **kwargs,
        )


def _download_project_optimized(
    api: Api,
    project_id,
    project_dir,
    datasets_whitelist=None,
    cache=None,
    progress_cb=None,
    only_image_tags=False,
    save_image_info=False,
    save_images=True,
    log_progress=True,
    images_ids: List[int] = None,
    **kwargs,
):

    skip_create_readme = kwargs.pop("skip_create_readme", False)

    project_info = api.project.get_info_by_id(project_id)
    project_id = project_info.id
    logger.info("Annotations are not cached (always download latest version from server)")
    project_fs = Project(project_dir, OpenMode.CREATE)
    meta = ProjectMeta.from_json(api.project.get_meta(project_id, with_settings=True))
    project_fs.set_meta(meta)

    if progress_cb is not None:
        log_progress = False

    for parents, dataset in api.dataset.tree(project_id):
        dataset_path = Dataset._get_dataset_path(dataset.name, parents)
        need_download = True

        if datasets_whitelist is not None and dataset.id not in datasets_whitelist:
            need_download = False

        if need_download is True:
            ds_progress = progress_cb
            if log_progress:
                ds_total = dataset.images_count
                if images_ids is not None:
                    ds_total = len(
                        api.image.get_list(
                            dataset.id,
                            filters=[{"field": "id", "operator": "in", "value": images_ids}],
                        )
                    )
                ds_progress = tqdm_sly(
                    desc="Downloading images from {!r}".format(dataset.name),
                    total=ds_total,
                )
            dataset_fs = project_fs.create_dataset(dataset.name, dataset_path)
            _download_dataset(
                api,
                dataset_fs,
                dataset.id,
                cache=cache,
                progress_cb=ds_progress,
                project_meta=meta,
                only_image_tags=only_image_tags,
                save_image_info=save_image_info,
                save_images=save_images,
                images_ids=images_ids,
            )
    if not skip_create_readme:
        try:
            create_readme(project_dir, project_id, api)
        except Exception as e:
            logger.info(f"There was an error while creating README: {e}")


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
    if name_split[1] == "":
        normalized_ext = ("." + ext).replace("..", ".")
        result = name + normalized_ext
        sly_image.validate_ext(result)
    else:
        result = name
    return result


def _download_dataset(
    api: Api,
    dataset: Dataset,
    dataset_id: int,
    cache=None,
    progress_cb=None,
    project_meta: ProjectMeta = None,
    only_image_tags=False,
    save_image_info=False,
    save_images=True,
    images_ids: List[int] = None,
):
    image_filters = None
    if images_ids is not None:
        image_filters = [{"field": "id", "operator": "in", "value": images_ids}]
    images = api.image.get_list(dataset_id, filters=image_filters)
    images_to_download = images
    if only_image_tags is True:
        if project_meta is None:
            raise ValueError("Project Meta is not defined")
        # pylint: disable=possibly-used-before-assignment
        id_to_tagmeta = project_meta.tag_metas.get_id_mapping()

    anns_progress = None
    if progress_cb is not None:
        anns_progress = tqdm_sly(
            desc="Downloading annotations from {!r}".format(dataset.name),
            total=len(images),
            leave=False,
        )
    # copy images from cache to task folder and download corresponding annotations
    if cache:
        (
            images_to_download,
            images_in_cache,
            images_cache_paths,
        ) = _split_images_by_cache(images, cache)
        if len(images_to_download) + len(images_in_cache) != len(images):
            raise RuntimeError("Error with images cache during download. Please contact support.")
        logger.info(
            f"Download dataset: {dataset.name}",
            extra={
                "total": len(images),
                "in cache": len(images_in_cache),
                "to download": len(images_to_download),
            },
        )
        if len(images_in_cache) > 0:
            img_cache_ids = [img_info.id for img_info in images_in_cache]

            if only_image_tags is False:
                with ApiContext(
                    api,
                    dataset_id=dataset_id,
                    project_meta=project_meta,
                ):
                    ann_info_list = api.annotation.download_batch(
                        dataset_id, img_cache_ids, anns_progress
                    )
                    img_name_to_ann = {ann.image_id: ann.annotation for ann in ann_info_list}
            else:
                img_name_to_ann = {}
                for image_info in images_in_cache:
                    # pylint: disable=possibly-used-before-assignment
                    tags = TagCollection.from_api_response(
                        image_info.tags,
                        project_meta.tag_metas,
                        id_to_tagmeta,
                    )
                    tmp_ann = Annotation(
                        img_size=(image_info.height, image_info.width), img_tags=tags
                    )
                    img_name_to_ann[image_info.id] = tmp_ann.to_json()
                if progress_cb is not None:
                    progress_cb(len(images_in_cache))

            for batch in batched(list(zip(images_in_cache, images_cache_paths)), batch_size=50):
                for img_info, img_cache_path in batch:
                    item_name = _maybe_append_image_extension(img_info.name, img_info.ext)
                    img_info_to_add = None
                    if save_image_info is True:
                        img_info_to_add = img_info
                    dataset.add_item_file(
                        item_name,
                        item_path=img_cache_path if save_images is True else None,
                        ann=img_name_to_ann[img_info.id],
                        _validate_item=False,
                        _use_hardlink=True,
                        item_info=img_info_to_add,
                    )
                if progress_cb is not None:
                    progress_cb(len(batch))

    # download images from server
    if len(images_to_download) > 0:
        # prepare lists for api methods
        img_ids = []
        img_paths = []
        for img_info in images_to_download:
            img_ids.append(img_info.id)
            img_paths.append(
                os.path.join(
                    dataset.item_dir,
                    _maybe_append_image_extension(img_info.name, img_info.ext),
                )
            )

        # download annotations
        if only_image_tags is False:
            ann_info_list = api.annotation.download_batch(dataset_id, img_ids, anns_progress)
            img_name_to_ann = {ann.image_id: ann.annotation for ann in ann_info_list}
        else:
            img_name_to_ann = {}
            for image_info in images_to_download:
                tags = TagCollection.from_api_response(
                    image_info.tags, project_meta.tag_metas, id_to_tagmeta
                )
                tmp_ann = Annotation(img_size=(image_info.height, image_info.width), img_tags=tags)
                img_name_to_ann[image_info.id] = tmp_ann.to_json()
            if progress_cb is not None:
                progress_cb(len(images_to_download))

        # download images and write to dataset
        for img_info_batch in batched(images_to_download):
            if save_images:
                images_ids_batch = [image_info.id for image_info in img_info_batch]
                images_nps = api.image.download_nps(
                    dataset_id, images_ids_batch, progress_cb=progress_cb
                )
            else:
                images_nps = [None] * len(img_info_batch)
            for index, image_np in enumerate(images_nps):
                img_info = img_info_batch[index]
                image_name = _maybe_append_image_extension(img_info.name, img_info.ext)

                dataset.add_item_np(
                    item_name=image_name,
                    img=image_np if save_images is True else None,
                    ann=img_name_to_ann[img_info.id],
                    img_info=img_info if save_image_info is True else None,
                )
        if cache is not None and save_images is True:
            img_hashes = [img_info.hash for img_info in images_to_download]
            cache.write_objects(img_paths, img_hashes)


def create_readme(
    project_dir: str,
    project_id: int,
    api: sly.Api,
) -> str:
    """Creates a README.md file using the template, adds general information
    about the project and creates a dataset structure section.

    :param project_dir: Path to the project directory.
    :type project_dir: str
    :param project_id: Project ID.
    :type project_id: int
    :param api: Supervisely API address and token.
    :type api: :class:`Api<supervisely.api.api.Api>`
    :return: Path to the created README.md file.
    :rtype: str

    :Usage example:

    .. code-block:: python

        import supervisely as sly

        api = sly.Api.from_env()

        project_id = 123
        project_dir = "/path/to/project"

        readme_path = sly.create_readme(project_dir, project_id, api)

        print(f"README.md file was created at {readme_path}")
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_path, "readme_template.md")
    with open(template_path, "r") as file:
        template = file.read()

    project_info = api.project.get_info_by_id(project_id)

    sly.fs.mkdir(project_dir)
    readme_path = os.path.join(project_dir, "README.md")

    template = template.replace("{{general_info}}", _project_info_md(project_info))

    template = template.replace(
        "{{dataset_structure_info}}", _dataset_structure_md(project_info, api)
    )

    template = template.replace(
        "{{dataset_description_info}}", _dataset_descriptions_md(project_info, api)
    )

    with open(readme_path, "w") as f:
        f.write(template)
    return readme_path


def _dataset_blob_structure_md(
    project_fs: Project,
    project_info: sly.ProjectInfo,
    entity_limit: Optional[int] = 2,
) -> str:
    """Creates a markdown string with the dataset structure of the project.
    Supports only images and videos projects.

    :project_fs: Project file system.
    :type project_fs: :class:`Project<supervisely.project.project.Project>`
    :param project_info: Project information.
    :type project_info: :class:`ProjectInfo<supervisely.project.project_info.ProjectInfo>`
    :param entity_limit: The maximum number of entities to display in the README.
    :type entity_limit: int, optional
    :return: Markdown string with the dataset structure of the project.
    :rtype: str
    """
    supported_project_types = [sly.ProjectType.IMAGES.value]
    if project_info.type not in supported_project_types:
        return ""

    entity_icons = {
        "images": " 🏞️ ",
        "blob_files": " 📦 ",
        "pkl_files": " 📄 ",
        "annotations": " 📝 ",
    }
    dataset_icon = " 📂 "
    folder_icon = " 📁 "

    result_md = f"🗂️ {project_info.name}<br>"

    # Add project-level blob files
    if os.path.exists(project_fs.blob_dir) and project_fs.blob_files:
        result_md += "┣" + folder_icon + f"{Project.blob_dir_name}<br>"
        blob_files = [entry.name for entry in os.scandir(project_fs.blob_dir) if entry.is_file()]

        for idx, blob_file in enumerate(blob_files):
            if idx == entity_limit and len(blob_files) > entity_limit:
                result_md += "┃ ┗ ... " + str(len(blob_files) - entity_limit) + " more<br>"
                break
            symbol = "┗" if idx == len(blob_files) - 1 or idx == entity_limit - 1 else "┣"
            result_md += "┃ " + symbol + entity_icons["blob_files"] + blob_file + "<br>"

    # Build a dataset hierarchy tree
    dataset_tree = {}
    root_datasets = []

    # First pass: create nodes for all datasets
    for dataset in project_fs.datasets:
        dataset_tree[dataset.directory] = {
            "dataset": dataset,
            "children": [],
            "parent_dir": os.path.dirname(dataset.directory) if dataset.parents else None,
        }

    # Second pass: build parent-child relationships
    for dir_path, node in dataset_tree.items():
        parent_dir = node["parent_dir"]
        if parent_dir in dataset_tree:
            dataset_tree[parent_dir]["children"].append(dir_path)
        else:
            root_datasets.append(dir_path)

    # Function to recursively render the dataset tree
    def render_tree(dir_path, prefix=""):
        nonlocal result_md
        node = dataset_tree[dir_path]
        dataset = node["dataset"]
        children = node["children"]

        # Create dataset display with proper path
        dataset_path = Dataset._get_dataset_path(dataset.name, dataset.parents)
        result_md += prefix + "┣" + dataset_icon + f"[{dataset.name}]({dataset_path})<br>"

        # Set indentation for dataset content
        content_prefix = prefix + "┃ "

        # Add pkl files at the dataset level
        offset_files = [
            entry.name
            for entry in os.scandir(dataset.directory)
            if entry.is_file() and entry.name.endswith(".pkl")
        ]

        if offset_files:
            for idx, pkl_file in enumerate(offset_files):
                last_file = idx == len(offset_files) - 1
                has_more_content = (
                    os.path.exists(dataset.img_dir) or os.path.exists(dataset.ann_dir) or children
                )
                symbol = "┗" if last_file and not has_more_content else "┣"
                result_md += content_prefix + symbol + entity_icons["pkl_files"] + pkl_file + "<br>"

        # Add img directory
        if os.path.exists(dataset.img_dir):
            has_ann_dir = os.path.exists(dataset.ann_dir)
            has_more_content = has_ann_dir or children
            symbol = "┣" if has_more_content else "┗"
            result_md += content_prefix + symbol + folder_icon + "img<br>"

            # Add image files
            entities = [entry.name for entry in os.scandir(dataset.img_dir) if entry.is_file()]
            entities = sorted(entities)
            selected_entities = entities[: min(len(entities), entity_limit)]

            img_prefix = content_prefix + "┃ "
            for idx, entity in enumerate(selected_entities):
                last_img = idx == len(selected_entities) - 1
                symbol = "┗" if last_img and len(entities) <= entity_limit else "┣"
                result_md += img_prefix + symbol + entity_icons["images"] + entity + "<br>"

            if len(entities) > entity_limit:
                result_md += img_prefix + "┗ ... " + str(len(entities) - entity_limit) + " more<br>"

        # Add ann directory
        if os.path.exists(dataset.ann_dir):
            has_more_content = bool(children)
            symbol = "┣"
            result_md += content_prefix + "┣" + folder_icon + "ann<br>"

            anns = [entry.name for entry in os.scandir(dataset.ann_dir) if entry.is_file()]
            anns = sorted(anns)

            # Try to match annotations with displayed images
            possible_anns = [f"{entity}.json" for entity in selected_entities]
            matched_anns = [pa for pa in possible_anns if pa in anns]

            # Add additional annotations if we haven't reached the limit
            if len(matched_anns) < min(entity_limit, len(anns)):
                for ann in anns:
                    if ann not in matched_anns and len(matched_anns) < entity_limit:
                        matched_anns.append(ann)

            ann_prefix = content_prefix + "┃ "
            for idx, ann in enumerate(matched_anns):
                last_ann = idx == len(matched_anns) - 1
                symbol = "┗" if last_ann and len(anns) <= entity_limit else "┣"
                result_md += ann_prefix + symbol + entity_icons["annotations"] + ann + "<br>"

            if len(anns) > entity_limit:
                result_md += ann_prefix + "┗ ... " + str(len(anns) - entity_limit) + " more<br>"

            if not has_more_content:
                result_md += content_prefix + "...<br>"
        # Recursively render child datasets
        for idx, child_dir in enumerate(children):
            render_tree(child_dir, content_prefix)

    # Start rendering from root datasets
    for root_dir in sorted(root_datasets):
        render_tree(root_dir)

    return result_md


def create_blob_readme(
    project_fs: Project,
    project_info: ProjectInfo,
    api: Api,
) -> str:
    """Creates a README.md file using the template, adds general information
    about the project and creates a dataset structure section.

    :param project_fs: Project file system.
    :type project_fs: :class:`Project<supervisely.project.project.Project>`
    :param project_info: Project information.
    :type project_info: :class:`ProjectInfo<supervisely.project.project_info.ProjectInfo>`
    :return: Path to the created README.md file.
    :rtype: str

    :Usage example:

    .. code-block:: python

        import supervisely as sly

        api = sly.Api.from_env()

        project_id = 123
        project_dir = "/path/to/project"

        readme_path = sly.create_readme(project_dir, project_id, api)

        print(f"README.md file was created at {readme_path}")
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_path, "readme_template.md")
    with open(template_path, "r") as file:
        template = file.read()

    readme_path = os.path.join(project_fs.directory, "README.md")

    template = template.replace("{{general_info}}", _project_info_md(project_info))

    template = template.replace(
        "{{dataset_structure_info}}", _dataset_blob_structure_md(project_fs, project_info)
    )
    template = template.replace(
        "{{dataset_description_info}}", _dataset_descriptions_md(project_info, api)
    )
    with open(readme_path, "w") as f:
        f.write(template)
    return readme_path


def _project_info_md(project_info: sly.ProjectInfo) -> str:
    """Creates a markdown string with general information about the project
    using the fields of the ProjectInfo NamedTuple.

    :param project_info: Project information.
    :type project_info: :class:`ProjectInfo<supervisely.project.project_info.ProjectInfo>`
    :return: Markdown string with general information about the project.
    :rtype: str
    """
    result_md = ""
    # Iterating over fields of a NamedTuple.
    for field in project_info._fields:
        value = getattr(project_info, field)
        if not value or not isinstance(value, (str, int)):
            # To avoid useless information in the README.
            continue
        result_md += f"\n**{snake_to_human(field)}:** {value}<br>"

    return result_md


def _dataset_structure_md(
    project_info: sly.ProjectInfo, api: sly.Api, entity_limit: Optional[int] = 4
) -> str:
    """Creates a markdown string with the dataset structure of the project.
    Supports only images and videos projects.

    :param project_info: Project information.
    :type project_info: :class:`ProjectInfo<supervisely.project.project_info.ProjectInfo>`
    :param api: Supervisely API address and token.
    :type api: :class:`Api<supervisely.api.api.Api>`
    :param entity_limit: The maximum number of entities to display in the README.
                        This is the limit for top level datasets and items in the dataset at the same time.
    :type entity_limit: int, optional
    :return: Markdown string with the dataset structure of the project.
    :rtype: str
    """
    # TODO: Add support for other project types.
    supported_project_types = [sly.ProjectType.IMAGES.value, sly.ProjectType.VIDEOS.value]
    if project_info.type not in supported_project_types:
        return ""

    list_functions = {
        "images": api.image.get_list,
        "videos": api.video.get_list,
    }
    entity_icons = {
        "images": " 🏞️ ",
        "videos": " 🎥 ",
        "blob_files": " 📦 ",
        "pkl_files": " 📄 ",
        "annotations": " 📝 ",
    }
    dataset_icon = " 📂 "
    list_function = list_functions[project_info.type]
    entity_icon = entity_icons[project_info.type]

    result_md = f"🗂️ {project_info.name}<br>"

    # Build a dataset hierarchy tree
    dataset_tree = {}
    root_datasets = []

    for parents, dataset_info in api.dataset.tree(project_info.id):
        level = len(parents)
        parent_id = dataset_info.parent_id

        if level == 0:  # Root dataset
            root_datasets.append(dataset_info)

        dataset_tree[dataset_info.id] = {
            "info": dataset_info,
            "path": Dataset._get_dataset_path(dataset_info.name, parents),
            "level": level,
            "parents": parents,
            "children": [],
        }

    # Connect parents with children
    for ds_id, ds_data in dataset_tree.items():
        parent_id = ds_data["info"].parent_id
        if parent_id in dataset_tree:
            dataset_tree[parent_id]["children"].append(ds_id)

    # Display only top entity_limit root datasets
    if len(root_datasets) > entity_limit:
        root_datasets = root_datasets[:entity_limit]
        result_md += f"(Showing only {entity_limit} top-level datasets)<br>"

    # Function to render a dataset and its children up to a certain depth
    def render_dataset(ds_id, current_depth=0, max_depth=2):
        if current_depth > max_depth:
            return

        ds_data = dataset_tree[ds_id]
        ds_info = ds_data["info"]
        basic_indent = "┃ " * current_depth

        # Render the dataset
        result_md.append(
            basic_indent + "┣ " + dataset_icon + f"[{ds_info.name}]({ds_data['path']})" + "<br>"
        )

        # Render items in the dataset
        entity_infos = list_function(ds_info.id)
        for idx, entity_info in enumerate(entity_infos):
            if idx == entity_limit:
                result_md.append(
                    basic_indent + "┃ ┗ ... " + str(len(entity_infos) - entity_limit) + " more<br>"
                )
                break
            symbol = "┗" if idx == len(entity_infos) - 1 else "┣"
            result_md.append(basic_indent + "┃ " + symbol + entity_icon + entity_info.name + "<br>")

        # Render children (limited to entity_limit)
        children = ds_data["children"]
        if len(children) > entity_limit:
            children = children[:entity_limit]
            result_md.append(basic_indent + f"┃ (Showing only {entity_limit} child datasets)<br>")

        for child_id in children:
            render_dataset(child_id, current_depth + 1, max_depth)

    # Render each root dataset
    result_md = [result_md]  # Convert to list for appending in the recursive function
    for root_ds in root_datasets:
        render_dataset(root_ds.id)

    return "".join(result_md)


def _dataset_descriptions_md(project_info: sly.ProjectInfo, api: sly.Api) -> str:
    """Creates a markdown string with dictionary of descriptions and custom data of datasets.
    :param project_info: Project information.
    :type project_info: :class:`ProjectInfo<supervisely.project.project_info.ProjectInfo>`
    :param api: Supervisely API address and token.
    :type api: :class:`Api<supervisely.api.api.Api>`
    :return: Markdown string with dictionary of descriptions and custom data of datasets.
    :rtype: str
    """

    data_found = False
    result_md = "All datasets in the project can have their own descriptions and custom data. You can add or edit the description and custom data of a dataset in the datasets list page. In this section, you can find this information for each dataset by dataset name (e.g. `ds1/ds2/ds3`, where `ds1` and `ds2` are parent datasets for `ds3` dataset).<br>"
    result_md += "\n\n```json\n{\n"
    for parents, dataset_info in api.dataset.tree(project_info.id):
        dataset_info = api.dataset.get_info_by_id(dataset_info.id)
        full_ds_name = "/".join(parents + [dataset_info.name])
        if dataset_info.description or dataset_info.custom_data:
            data_found = True
            result_md += f'  "{full_ds_name}": {{\n'
            if dataset_info.description:
                result_md += f'    "description": "{dataset_info.description}",\n'
            if dataset_info.custom_data:
                formated_custom_data = json.dumps(dataset_info.custom_data, indent=4)
                formated_custom_data = formated_custom_data.replace("\n", "\n    ")
                result_md += f'    "custom_data": {formated_custom_data}\n'
            result_md += "  },\n"
    result_md += "}\n```"
    if not data_found:
        result_md = "_No dataset descriptions or custom data found in the project._"
    return result_md


async def _download_project_async(
    api: sly.Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: bool = True,
    semaphore: asyncio.Semaphore = None,
    only_image_tags: Optional[bool] = False,
    save_image_info: Optional[bool] = False,
    save_images: Optional[bool] = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    save_image_meta: Optional[bool] = False,
    images_ids: Optional[List[int]] = None,
    resume_download: Optional[bool] = False,
    **kwargs,
):
    """
    Download image project to the local directory asynchronously.
    Uses queue and semaphore to control the number of parallel downloads.
    Every image goes through size check to decide if it should be downloaded in bulk or one by one.
    Checked images are split into two lists: small and large. Small images are downloaded in bulk, large images are downloaded one by one.
    As soon as the task is created, it is put into the queue. Workers take tasks from the queue and execute them.

    """
    # to switch between single and bulk download
    switch_size = kwargs.get("switch_size", 1.28 * 1024 * 1024)
    # batch size for bulk download
    batch_size = kwargs.get("batch_size", 100)
    # control whether to download blob files
    download_blob_files = kwargs.get("download_blob_files", False)
    # control whether to create README file
    skip_create_readme = kwargs.get("skip_create_readme", False)

    if semaphore is None:
        semaphore = api.get_default_semaphore()

    dataset_ids = set(dataset_ids) if (dataset_ids is not None) else None
    project_fs = None
    meta = ProjectMeta.from_json(api.project.get_meta(project_id, with_settings=True))
    if os.path.exists(dest_dir) and resume_download:
        dump_json_file(meta.to_json(), os.path.join(dest_dir, "meta.json"))
        try:
            project_fs = Project(dest_dir, OpenMode.READ)
        except RuntimeError as e:
            if "Project is empty" in str(e):
                clean_dir(dest_dir)
                project_fs = None
            else:
                raise
    if project_fs is None:
        project_fs = Project(dest_dir, OpenMode.CREATE)
    project_fs.set_meta(meta)

    if progress_cb is not None:
        log_progress = False

    id_to_tagmeta = None
    if only_image_tags is True:
        id_to_tagmeta = meta.tag_metas.get_id_mapping()

    existing_datasets = {dataset.path: dataset for dataset in project_fs.datasets}
    for parents, dataset in api.dataset.tree(project_id):
        dataset_path = Dataset._get_dataset_path(dataset.name, parents)
        dataset_id = dataset.id
        if dataset_ids is not None and dataset_id not in dataset_ids:
            continue

        if dataset_path in existing_datasets:
            dataset_fs = existing_datasets[dataset_path]
        else:
            dataset_fs = project_fs.create_dataset(dataset.name, dataset_path)

        force_metadata_for_links = False
        if save_images is False and only_image_tags is True:
            force_metadata_for_links = True
        all_images = api.image.get_list_generator_async(
            dataset_id, force_metadata_for_links=force_metadata_for_links, dataset_info=dataset
        )
        small_images = []
        large_images = []
        dataset_images = []
        blob_files_to_download = {}
        blob_images = []

        sly.logger.info("Calculating images to download...", extra={"dataset": dataset.name})
        async for image_batch in all_images:
            for image in image_batch:
                if images_ids is None or image.id in images_ids:
                    dataset_images.append(image)
                    # Check for images with blob offsets

                    if download_blob_files and image.related_data_id is not None:
                        blob_files_to_download[image.related_data_id] = image.download_id
                        blob_images.append(image)
                    elif image.size is not None and image.size < switch_size:
                        small_images.append(image)
                    else:
                        large_images.append(image)

        ds_progress = progress_cb
        if log_progress is True:
            ds_progress = tqdm_sly(
                desc="Downloading images from {!r}".format(dataset.name),
                total=len(small_images) + len(large_images) + len(blob_images),
                leave=False,
            )

        with ApiContext(
            api,
            project_id=project_id,
            dataset_id=dataset_id,
            project_meta=meta,
        ):

            async def check_items(check_list: List[sly.ImageInfo]):
                to_download = []
                for image in check_list:
                    try:
                        existing = dataset_fs.get_item_info(image.name)
                    except:
                        to_download.append(image)
                    else:
                        if existing.updated_at != image.updated_at:
                            to_download.append(image)
                        elif ds_progress is not None:
                            ds_progress(1)
                return to_download

            async def run_tasks_with_semaphore_control(task_list: list, delay=0.05):
                """
                Execute tasks with semaphore control - create tasks only as semaphore permits become available.
                task_list - list of coroutines or callables that create tasks
                """
                random.shuffle(task_list)
                running_tasks = set()
                max_concurrent = getattr(semaphore, "_value", 10)

                task_iter = iter(task_list)
                completed_count = 0

                while True:
                    # Add new tasks while we have capacity
                    while len(running_tasks) < max_concurrent:
                        try:
                            task_gen = next(task_iter)
                            if callable(task_gen):
                                task = asyncio.create_task(task_gen())
                            else:
                                task = asyncio.create_task(task_gen)
                            running_tasks.add(task)
                            await asyncio.sleep(delay)
                        except StopIteration:
                            break

                    if not running_tasks:
                        break

                    # Wait for at least one task to complete
                    done, running_tasks = await asyncio.wait(
                        running_tasks, return_when=asyncio.FIRST_COMPLETED
                    )

                    # Process completed tasks
                    for task in done:
                        completed_count += 1
                        try:
                            await task
                        except Exception as e:
                            logger.error(f"Task error: {e}")

                    # Clear the done set - this should be enough for memory cleanup
                    done.clear()

                logger.debug(
                    f"{completed_count} tasks have been completed for dataset ID: {dataset.id}, Name: {dataset.name}"
                )
                return completed_count

            # Download blob files if required
            if download_blob_files and len(blob_files_to_download) > 0:
                blob_paths = []
                download_ids = []
                # Process each blob file
                for blob_file_id, download_id in blob_files_to_download.items():
                    if blob_file_id not in project_fs.blob_files:
                        # Download the blob file
                        blob_paths.append(os.path.join(project_fs.blob_dir, f"{blob_file_id}.tar"))
                        download_ids.append(download_id)
                await api.image.download_blob_files_async(
                    project_id=project_id,
                    download_ids=download_ids,
                    paths=blob_paths,
                    semaphore=semaphore,
                    log_progress=(True if log_progress or progress_cb is not None else False),
                )
                for blob_file_id, download_id in blob_files_to_download.items():
                    project_fs.add_blob_file(blob_file_id)

                    # Process blob image offsets
                    offsets_file_name = f"{blob_file_id}{OFFSETS_PKL_SUFFIX}"
                    offsets_file_path = os.path.join(dataset_fs.directory, offsets_file_name)

                    total_offsets_count = 0  # for logging
                    current_batch = []
                    for img in blob_images:
                        if img.related_data_id == blob_file_id:
                            blob_image_info = BlobImageInfo(
                                name=img.name,
                                offset_start=img.offset_start,
                                offset_end=img.offset_end,
                            )
                            current_batch.append(blob_image_info)
                        if len(current_batch) >= OFFSETS_PKL_BATCH_SIZE:
                            BlobImageInfo.dump_to_pickle(current_batch, offsets_file_path)
                            total_offsets_count += len(current_batch)
                            current_batch = []
                    if len(current_batch) > 0:
                        BlobImageInfo.dump_to_pickle(current_batch, offsets_file_path)
                        total_offsets_count += len(current_batch)
                    if total_offsets_count > 0:
                        logger.debug(
                            f"Saved {total_offsets_count} image offsets for {blob_file_id} to {offsets_file_path} in {(total_offsets_count + OFFSETS_PKL_BATCH_SIZE - 1) // OFFSETS_PKL_BATCH_SIZE} batches"
                        )
                    offset_tasks = []
                    # Download annotations for images with offsets
                    for offsets_batch in batched(blob_images, batch_size=batch_size):
                        offset_task = _download_project_items_batch_async(
                            api=api,
                            dataset_id=dataset_id,
                            img_infos=offsets_batch,
                            meta=meta,
                            dataset_fs=dataset_fs,
                            id_to_tagmeta=id_to_tagmeta,
                            semaphore=semaphore,
                            save_images=False,
                            save_image_info=save_image_info,
                            only_image_tags=only_image_tags,
                            progress_cb=ds_progress,
                        )
                        offset_tasks.append(offset_task)
                    await run_tasks_with_semaphore_control(offset_tasks, 0.05)

            tasks = []
            if resume_download is True:
                sly.logger.info("Checking existing images...", extra={"dataset": dataset.name})
                # Check which images need to be downloaded
                small_images = await check_items(small_images)
                large_images = await check_items(large_images)

            # If only one small image, treat it as a large image for efficiency
            if len(small_images) == 1:
                large_images.append(small_images.pop())

            # Create batch download tasks
            sly.logger.debug(
                f"Downloading {len(small_images)} small images in batch number {len(small_images) // batch_size}...",
                extra={"dataset": dataset.name},
            )
            for images_batch in batched(small_images, batch_size=batch_size):
                task = _download_project_items_batch_async(
                    api=api,
                    dataset_id=dataset_id,
                    img_infos=images_batch,
                    meta=meta,
                    dataset_fs=dataset_fs,
                    id_to_tagmeta=id_to_tagmeta,
                    semaphore=semaphore,
                    save_images=save_images,
                    save_image_info=save_image_info,
                    only_image_tags=only_image_tags,
                    progress_cb=ds_progress,
                )
                tasks.append(task)

            # Create individual download tasks for large images
            sly.logger.debug(
                f"Downloading {len(large_images)} large images one by one...",
                extra={"dataset": dataset.name},
            )
            for image in large_images:
                task = _download_project_item_async(
                    api=api,
                    img_info=image,
                    meta=meta,
                    dataset_fs=dataset_fs,
                    id_to_tagmeta=id_to_tagmeta,
                    semaphore=semaphore,
                    save_images=save_images,
                    save_image_info=save_image_info,
                    only_image_tags=only_image_tags,
                    progress_cb=ds_progress,
                )
                tasks.append(task)

            await run_tasks_with_semaphore_control(tasks)

        if save_image_meta:
            meta_dir = dataset_fs.meta_dir
            for image_info in dataset_images:
                if image_info.meta:
                    sly.fs.mkdir(meta_dir)
                    sly.json.dump_json_file(
                        image_info.meta, dataset_fs.get_item_meta_path(image_info.name)
                    )

        # delete redundant items
        items_names_set = set([img.name for img in dataset_images])
        for item_name in dataset_fs.get_items_names():
            if item_name not in items_names_set:
                dataset_fs.delete_item(item_name)
    if not skip_create_readme:
        try:
            if download_blob_files:
                project_info = api.project.get_info_by_id(project_id)
                create_blob_readme(project_fs=project_fs, project_info=project_info, api=api)
            else:
                create_readme(dest_dir, project_id, api)
        except Exception as e:
            logger.info(f"There was an error while creating README: {e}")


async def _download_project_item_async(
    api: sly.Api,
    img_info: sly.ImageInfo,
    meta: ProjectMeta,
    dataset_fs: Dataset,
    id_to_tagmeta: Dict[int, sly.TagMeta],
    semaphore: asyncio.Semaphore,
    save_images: bool,
    save_image_info: bool,
    only_image_tags: bool,
    progress_cb: Optional[Callable],
) -> None:
    """Download image and annotation from Supervisely API and save it to the local filesystem.
    Uses parameters from the parent function _download_project_async.
    Optimized version - uses streaming only for large images (>5MB) to avoid performance degradation.
    """

    # Prepare annotation first (small data)
    if only_image_tags is False:
        ann_info = await api.annotation.download_async(
            img_info.id,
            semaphore=semaphore,
            force_metadata_for_links=not save_images,
        )
        ann_json = ann_info.annotation
        try:
            tmp_ann = Annotation.from_json(ann_json, meta)
        except Exception:
            logger.error(f"Error while deserializing annotation for image with ID: {img_info.id}")
            raise
        if None in tmp_ann.img_size:
            tmp_ann = tmp_ann.clone(img_size=(img_info.height, img_info.width))
            ann_json = tmp_ann.to_json()
    else:
        tags = TagCollection.from_api_response(
            img_info.tags,
            meta.tag_metas,
            id_to_tagmeta,
        )
        tmp_ann = Annotation(img_size=(img_info.height, img_info.width), img_tags=tags)
        ann_json = tmp_ann.to_json()

    # Handle image download - choose method based on estimated size
    if save_images:
        # Estimate size threshold: 5MB for streaming to avoid performance degradation
        size_threshold_for_streaming = 5 * 1024 * 1024  # 5MB
        estimated_size = getattr(img_info, "size", 0) or (
            img_info.height * img_info.width * 3 if img_info.height and img_info.width else 0
        )

        if estimated_size > size_threshold_for_streaming:
            # Use streaming for large images only
            sly.logger.trace(
                f"Downloading large image in streaming mode: {img_info.size / 1024 / 1024:.1f}MB"
            )

            # Clean up existing item first
            dataset_fs.delete_item(img_info.name)

            final_path = dataset_fs.generate_item_path(img_info.name)
            temp_path = final_path + ".tmp"
            await api.image.download_path_async(
                img_info.id, temp_path, semaphore=semaphore, check_hash=True
            )

            # Get dimensions if needed
            if None in [img_info.height, img_info.width]:
                # Use PIL directly on the file - it will only read the minimal header needed
                with PILImage.open(temp_path) as image:
                    width, height = image.size
                img_info = img_info._replace(height=height, width=width)

            # Update annotation with correct dimensions if needed
            if None in tmp_ann.img_size:
                tmp_ann = tmp_ann.clone(img_size=(img_info.height, img_info.width))
                ann_json = tmp_ann.to_json()

            # os.rename is atomic and will overwrite the destination if it exists
            os.rename(temp_path, final_path)

            # For streaming, we save directly to filesystem, so use add_item_raw_bytes_async with None
            await dataset_fs.add_item_raw_bytes_async(
                item_name=img_info.name,
                item_raw_bytes=None,  # Image already saved to disk
                ann=ann_json,
                img_info=img_info if save_image_info is True else None,
            )
        else:
            sly.logger.trace(f"Downloading large image: {img_info.size / 1024 / 1024:.1f}MB")
            # Use fast in-memory download for small images
            img_bytes = await api.image.download_bytes_single_async(
                img_info.id, semaphore=semaphore, check_hash=True
            )

            if None in [img_info.height, img_info.width]:
                width, height = sly.image.get_size_from_bytes(img_bytes)
                img_info = img_info._replace(height=height, width=width)

            # Update annotation with correct dimensions if needed
            if None in tmp_ann.img_size:
                tmp_ann = tmp_ann.clone(img_size=(img_info.height, img_info.width))
                ann_json = tmp_ann.to_json()

            # Clean up existing item first, then save new one
            dataset_fs.delete_item(img_info.name)
            await dataset_fs.add_item_raw_bytes_async(
                item_name=img_info.name,
                item_raw_bytes=img_bytes,
                ann=ann_json,
                img_info=img_info if save_image_info is True else None,
            )
    else:
        dataset_fs.delete_item(img_info.name)
        await dataset_fs.add_item_raw_bytes_async(
            item_name=img_info.name,
            item_raw_bytes=None,
            ann=ann_json,
            img_info=img_info if save_image_info is True else None,
        )

    if progress_cb is not None:
        progress_cb(1)
    logger.debug(f"Single project item has been downloaded. Semaphore state: {semaphore._value}")


async def _download_project_items_batch_async(
    api: sly.Api,
    dataset_id: int,
    img_infos: List[sly.ImageInfo],
    meta: ProjectMeta,
    dataset_fs: Dataset,
    id_to_tagmeta: Dict[int, sly.TagMeta],
    semaphore: asyncio.Semaphore,
    save_images: bool,
    save_image_info: bool,
    only_image_tags: bool,
    progress_cb: Optional[Callable],
):
    """
    Download images and annotations from Supervisely API and save them to the local filesystem.
    Uses parameters from the parent function _download_project_async.
    It is used for batch download of images and annotations with the bulk download API methods.

    IMPORTANT: The total size of all images in a batch must not exceed 130MB, and the size of each image must not exceed 1.28MB.
    """
    img_ids = [img_info.id for img_info in img_infos]
    img_ids_to_info = {img_info.id: img_info for img_info in img_infos}

    sly.logger.trace(f"Downloading {len(img_infos)} images in batch mode.")
    # Download annotations first
    if only_image_tags is False:
        ann_infos = await api.annotation.download_bulk_async(
            dataset_id,
            img_ids,
            semaphore=semaphore,
            force_metadata_for_links=not save_images,
        )
        id_to_annotation = {}
        for img_info, ann_info in zip(img_infos, ann_infos):
            try:
                tmp_ann = Annotation.from_json(ann_info.annotation, meta)
                if None in tmp_ann.img_size:
                    tmp_ann = tmp_ann.clone(img_size=(img_info.height, img_info.width))
                id_to_annotation[img_info.id] = tmp_ann.to_json()
            except Exception:
                logger.error(
                    f"Error while deserializing annotation for image with ID: {img_info.id}"
                )
                raise
    else:
        id_to_annotation = {}
        for img_info in img_infos:
            tags = TagCollection.from_api_response(
                img_info.tags,
                meta.tag_metas,
                id_to_tagmeta,
            )
            tmp_ann = Annotation(img_size=(img_info.height, img_info.width), img_tags=tags)
            id_to_annotation[img_info.id] = tmp_ann.to_json()

    if save_images:
        async for img_id, img_bytes in api.image.download_bytes_generator_async(
            dataset_id=dataset_id, img_ids=img_ids, semaphore=semaphore, check_hash=True
        ):
            img_info = img_ids_to_info.get(img_id)
            if img_info is None:
                continue

            if None in [img_info.height, img_info.width]:
                width, height = sly.image.get_size_from_bytes(img_bytes)
                img_info = img_info._replace(height=height, width=width)

                # Update annotation if needed - use pop to get and remove at the same time
                ann_json = id_to_annotation.pop(img_id, None)
                if ann_json is not None:
                    try:
                        tmp_ann = Annotation.from_json(ann_json, meta)
                        if None in tmp_ann.img_size:
                            tmp_ann = tmp_ann.clone(img_size=(img_info.height, img_info.width))
                        ann_json = tmp_ann.to_json()
                    except Exception:
                        pass
            else:
                ann_json = id_to_annotation.pop(img_id, None)

            dataset_fs.delete_item(img_info.name)
            await dataset_fs.add_item_raw_bytes_async(
                item_name=img_info.name,
                item_raw_bytes=img_bytes,
                ann=ann_json,
                img_info=img_info if save_image_info is True else None,
            )

            if progress_cb is not None:
                progress_cb(1)
    else:
        for img_info in img_infos:
            dataset_fs.delete_item(img_info.name)
            ann_json = id_to_annotation.pop(img_info.id, None)
            await dataset_fs.add_item_raw_bytes_async(
                item_name=img_info.name,
                item_raw_bytes=None,
                ann=ann_json,
                img_info=img_info if save_image_info is True else None,
            )
            if progress_cb is not None:
                progress_cb(1)

    # Clear dictionaries and force GC for large batches only
    batch_size = len(img_infos)
    id_to_annotation.clear()
    img_ids_to_info.clear()

    if batch_size > 50:  # Only for large batches
        gc.collect()

    logger.debug(f"Batch of project items has been downloaded. Semaphore state: {semaphore._value}")


DatasetDict = Project.DatasetDict
