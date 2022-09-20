# coding: utf-8

from __future__ import annotations
import shutil
from collections import namedtuple
import os
import json
from enum import Enum
from typing import List, Dict, Optional, NamedTuple, Tuple, Union, Callable
import random
import numpy as np

from supervisely.annotation.annotation import Annotation, ANN_EXT, TagCollection
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.api.project_api import ProjectInfo
from supervisely.collection.key_indexed_collection import (
    KeyIndexedCollection,
    KeyObject,
)
from supervisely.imaging import image as sly_image
from supervisely.io.fs import (
    list_files,
    list_files_recursively,
    mkdir,
    copy_file,
    get_subdirs,
    dir_exists,
    dir_empty,
    silent_remove,
)
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress
from supervisely._utils import batched, is_development, abs_url
from supervisely.io.fs import file_exists, ensure_base_path, get_file_name
from supervisely.api.api import Api
from supervisely.sly_logger import logger
from supervisely.io.fs_cache import FileCache
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.rectangle import Rectangle

# @TODO: rename img_path to item_path
ItemPaths = namedtuple("ItemPaths", ["img_path", "ann_path"])
ItemInfo = namedtuple("ItemInfo", ["dataset_name", "name", "img_path", "ann_path"])


class OpenMode(Enum):
    """
    Defines the mode of using the :class:`Project<supervisely.project.project.Project>` and :class:`Dataset<supervisely.project.project.Dataset>`.
    """

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
    """
    Dataset is where your labeled and unlabeled images, videos and other files live. :class:`Dataset<Dataset>` object is immutable.

    :param directory: Path to dataset directory.
    :type directory: str
    :param mode: Determines working mode for the given dataset.
    :type mode: OpenMode
    :Usage example:

     .. code-block:: python

         from supervisely.project.project import Dataset, OpenMode
         dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
         ds = Dataset(dataset_path, OpenMode.READ)
    """

    item_dir_name = "img"
    annotation_class = Annotation

    def __init__(self, directory: str, mode: OpenMode):
        """
        :param directory: path to the directory where the data set will be saved or where it will be loaded from
        :param mode: OpenMode class object which determines in what mode to work with the dataset
        """
        if type(mode) is not OpenMode:
            raise TypeError(
                "Argument 'mode' has type {!r}. Correct type is OpenMode".format(
                    type(mode)
                )
            )

        self._directory = directory
        self._item_to_ann = {}  # item file name -> annotation file name

        project_dir, ds_name = os.path.split(directory.rstrip("/"))
        self._project_dir = project_dir
        self._name = ds_name

        if mode is OpenMode.READ:
            self._read()
        else:
            self._create()

    @property
    def name(self) -> str:
        """
        Dataset name.

        :return: Dataset Name
        :rtype: :`class`: str
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)
            print(dataset.name)
            # Output: "ds1"
        """
        return self._name

    def key(self):
        return self.name

    @property
    def directory(self) -> str:
        """
        Path to Dataset directory.

        :return: Path to dataset directory
        :rtype: :class:`str`

        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.directory)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1'
        """
        return self._directory

    @property
    def item_dir(self) -> str:
        """
        Path to the dataset items directory.

        :return: Path to the dataset directory with items
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.img_dir)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/img'
        """
        return self.img_dir

    @property
    def img_dir(self) -> str:
        """
        Path to the dataset images directory.

        :return: Path to the dataset directory with images
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.img_dir)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/img'
        """
        # @TODO: deprecated method, should be private and be renamed in future
        return os.path.join(self.directory, self.item_dir_name)

    @property
    def ann_dir(self) -> str:
        """
        Path to the dataset annotations directory.

        :return: Path to the dataset directory with annotations
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.ann_dir)
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/ann'
        """
        return os.path.join(self.directory, "ann")

    @property
    def img_info_dir(self):
        return os.path.join(self.directory, "img_info")

    @property
    def seg_dir(self):
        return os.path.join(self.directory, "seg")

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
        Consistency checks. Every image must have an annotation, and the correspondence must be one to one.
        If not - it generate exception error.
        """
        if not dir_exists(self.item_dir):
            raise FileNotFoundError(
                "Item directory not found: {!r}".format(self.item_dir)
            )
        if not dir_exists(self.ann_dir):
            raise FileNotFoundError(
                "Annotation directory not found: {!r}".format(self.ann_dir)
            )

        raw_ann_paths = list_files(self.ann_dir, [ANN_EXT])
        img_paths = list_files(self.item_dir, filter_fn=self._has_valid_ext)

        raw_ann_names = set(os.path.basename(path) for path in raw_ann_paths)
        img_names = [os.path.basename(path) for path in img_paths]

        if len(img_names) == 0 and len(raw_ann_names) == 0:
            raise RuntimeError("Dataset {!r} is empty".format(self.name))

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

    def _create(self):
        """
        Creates a leaf directory and all intermediate ones for items and annatations.
        """
        mkdir(self.ann_dir)
        mkdir(self.item_dir)

    def get_items_names(self) -> list:
        return list(self._item_to_ann.keys())

    def item_exists(self, item_name: str) -> bool:
        """
        Checks if given item name belongs to the dataset.

        :param item_name: Item name.
        :type item_name: str
        :return: True if item exist, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            ds.item_exists("IMG_0748")      # False
            ds.item_exists("IMG_0748.jpeg") # True
        """
        return item_name in self._item_to_ann

    def get_item_path(self, item_name: str) -> str:
        """
        Path to the given item.

        :param item_name: Item name.
        :type item_name: str
        :return: Path to the given item
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            ds.get_item_path("IMG_0748")
            # Output: RuntimeError: Item IMG_0748 not found in the project.

            ds.get_item_path("IMG_0748.jpeg")
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/img/IMG_0748.jpeg'
        """
        return self.get_img_path(item_name)

    def get_img_path(self, item_name: str) -> str:
        """
        Path to the given image.

        :param item_name: Image name.
        :type item_name: str
        :return: Path to the given image
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            ds.get_img_path("IMG_0748")
            # Output: RuntimeError: Item IMG_0748 not found in the project.

            ds.get_img_path("IMG_0748.jpeg")
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/ann/IMG_0748.jpeg.json'
        """
        # @TODO: deprecated method, should be private and be renamed in future
        if not self.item_exists(item_name):
            raise RuntimeError("Item {} not found in the project.".format(item_name))

        return os.path.join(self.item_dir, item_name)

    def get_ann(self, item_name, project_meta: ProjectMeta) -> Annotation:
        ann_path = self.get_ann_path(item_name)
        return Annotation.load_json_file(ann_path, project_meta)

    def get_ann_path(self, item_name: str) -> str:
        """
        Path to the given annotation.

        :param item_name: Annotation name.
        :type item_name: str
        :return: Path to the given annotation
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            ds.get_ann_path("IMG_0748")
            # Output: RuntimeError: Item IMG_0748 not found in the project.

            ds.get_ann_path("IMG_0748.jpeg")
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/ann/IMG_0748.jpeg.json'
        """
        ann_path = self._item_to_ann.get(item_name, None)
        if ann_path is None:
            raise RuntimeError("Item {} not found in the project.".format(item_name))

        return os.path.join(self.ann_dir, ann_path)

    def get_img_info_path(self, item_name: str) -> str:
        ann_path = self._item_to_ann.get(item_name, None)
        if ann_path is None:
            raise RuntimeError("Item {} not found in the project.".format(item_name))

        return os.path.join(self.img_info_dir, ann_path)

    def get_image_info(self, item_name: str) -> NamedTuple:
        """
        Information for Image with given name.

        :param item_name: Image name.
        :type item_name: str
        :return: Image with information for the given Dataset
        :rtype: :class:`NamedTuple`
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            info = ds.get_image_info("IMG_0748.jpeg")
        """
        img_info_path = self.get_img_info_path(item_name)
        image_info_dict = load_json_file(img_info_path)
        ImageInfo = namedtuple("ImageInfo", image_info_dict)
        info = ImageInfo(**image_info_dict)
        return info

    def get_seg_path(self, item_name: str) -> str:
        ann_path = self._item_to_ann.get(item_name, None)
        if ann_path is None:
            raise RuntimeError("Item {} not found in the project.".format(item_name))
        seg_path = os.path.join(self.seg_dir, f"{item_name}.png")
        return seg_path

    def add_item_file(
        self,
        item_name: str,
        item_path: str,
        ann: Optional[Union[Annotation, str]] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
        img_info: Optional[NamedTuple] = None,
    ) -> None:
        """
        Adds given item file to dataset items directory, and adds given annotation to dataset ann directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: str
        :param item_path: Path to the item.
        :type item_path: str
        :param ann: Annotation object or path to annotation.json file.
        :type ann: Annotation or str, optional
        :param _validate_item: Checks input files format.
        :type _validate_item: bool, optional
        :param _use_hardlink: If True creates a hardlink pointing to src named dst, otherwise don't.
        :type _use_hardlink: bool, optional
        :param img_info: NamedTuple ImageInfo containing information about Image.
        :type img_info: NamedTuple, optional
        :return: None
        :rtype: :class:`NoneType`
        :raises: :class:`Exception` if item_name already exists in dataset or item name has unsupported extension.
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            ann = "/home/admin/work/supervisely/projects/Test/IMG_8888.jpeg.json"
            ds.add_item_file("IMG_777.jpeg", "/home/admin/work/supervisely/projects/Test/IMG_8888.jpeg", ann=ann)
        """
        if item_path is None and ann is None and img_info is None:
            raise RuntimeError("No item_path or ann or img_info provided.")

        self._add_item_file(
            item_name,
            item_path,
            _validate_item=_validate_item,
            _use_hardlink=_use_hardlink,
        )
        self._add_ann_by_type(item_name, ann)
        self._add_img_info(item_name, img_info)

    def add_item_np(
        self,
        item_name: str,
        img: np.ndarray,
        ann: Optional[Union[Annotation, str]] = None,
        img_info: Optional[NamedTuple] = None,
    ) -> None:
        """
        Adds given numpy matrix as an image to dataset items directory, and adds given annotation to dataset ann directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: str
        :param img: numpy Image matrix in RGB format.
        :type img: np.ndarray
        :param ann: Annotation object or path to annotation.json file.
        :type ann: Annotation or str, optional
        :param img_info: NamedTuple ImageInfo containing information about Image.
        :type img_info: NamedTuple, optional
        :return: None
        :rtype: :class:`NoneType`
        :raises: :class:`Exception` if item_name already exists in dataset or item name has unsupported extension
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            img_path = "/home/admin/Pictures/Clouds.jpeg"
            img_np = sly.image.read(img_path)
            ds.add_item_np("IMG_050.jpeg", img_np)
        """
        if img is None and ann is None and img_info is None:
            raise RuntimeError("No img or ann or img_info provided.")

        self._add_img_np(item_name, img)
        self._add_ann_by_type(item_name, ann)
        self._add_img_info(item_name, img_info)

    def add_item_raw_bytes(
        self,
        item_name: str,
        item_raw_bytes: bytes,
        ann: Optional[Union[Annotation, str]] = None,
        img_info: Optional[NamedTuple] = None,
    ) -> None:
        """
        Adds given binary object as an image to dataset items directory, and adds given annotation to dataset ann directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: str
        :param item_raw_bytes: Binary object.
        :type item_raw_bytes: bytes
        :param ann: Annotation object or path to annotation.json file.
        :type ann: Annotation or str, optional
        :param img_info: NamedTuple ImageInfo containing information about Image.
        :type img_info: NamedTuple, optional
        :return: None
        :rtype: :class:`NoneType`
        :raises: :class:`Exception` if item_name already exists in dataset or item name has unsupported extension
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            img_path = "/home/admin/Pictures/Clouds.jpeg"
            img_np = sly.image.read(img_path)
            img_bytes = sly.image.write_bytes(img_np, "jpeg")
            ds.add_item_np("IMG_050.jpeg", img_bytes)
        """
        if item_raw_bytes is None and ann is None and img_info is None:
            raise RuntimeError("No item_raw_bytes or ann or img_info provided.")

        self._add_item_raw_bytes(item_name, item_raw_bytes)
        self._add_ann_by_type(item_name, ann)
        self._add_img_info(item_name, img_info)

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
        Add given annatation to dataset annotations dir and to dictionary items: item file name -> annotation file name
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

    def _add_img_info(self, item_name, img_info=None):
        if img_info is None:
            return

        dst_info_path = self.get_img_info_path(item_name)
        ensure_base_path(dst_info_path)
        if type(img_info) is dict:
            dump_json_file(img_info, dst_info_path, indent=4)
        elif type(img_info) is str and os.path.isfile(img_info):
            shutil.copy(img_info, dst_info_path)
        else:
            # ImgInfo named tuple
            dump_json_file(img_info._asdict(), dst_info_path, indent=4)

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
            raise RuntimeError(
                "Item name {!r} has unsupported extension.".format(item_name)
            )

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
        dst_img_path = os.path.join(self.item_dir, item_name)
        with open(dst_img_path, "wb") as fout:
            fout.write(item_raw_bytes)
        self._validate_added_item_or_die(dst_img_path)

    def generate_item_path(self, item_name: str) -> str:
        """
        Generates full path to the given item.

        :param item_name: Item name.
        :type item_name: str
        :return: Full path to the given item
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            print(ds.generate_item_path("IMG_0748.jpeg"))
            # Output: '/home/admin/work/supervisely/projects/lemons_annotated/ds1/img/IMG_0748.jpeg'
        """
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

    def _add_item_file(
        self, item_name, item_path, _validate_item=True, _use_hardlink=False
    ):
        if item_path is None:
            return

        self._add_img_file(item_name, item_path, _validate_item, _use_hardlink)

    def _add_img_file(
        self, item_name, img_path, _validate_img=True, _use_hardlink=False
    ):
        """
        Add given item file to dataset items directory. Generate exception error if item_name already exists in dataset
        or item name has unsupported extension
        :param item_name: str
        :param img_path: str
        :param _validate_img: bool
        :param _use_hardlink: bool
        """
        # @TODO: deprecated method, should be private and be (refactored, renamed) in future
        self._check_add_item_name(item_name)
        dst_img_path = os.path.join(self.item_dir, item_name)
        if (
            img_path != dst_img_path and img_path is not None
        ):  # used only for agent + api during download project + None to optimize internal usage
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
        :type item_name: str
        :param ann: Annotation object.
        :type ann: Annotation
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            height, width = 500, 700
            ann = sly.Annotation((height, width))
            ds.set_ann("IMG_0748.jpeg", ann)
        """
        if type(ann) is not self.annotation_class:
            raise TypeError(
                "Type of 'ann' have to be Annotation, not a {}".format(type(ann))
            )
        dst_ann_path = self.get_ann_path(item_name)
        dump_json_file(ann.to_json(), dst_ann_path, indent=4)

    def set_ann_file(self, item_name: str, ann_path: str) -> None:
        """
        Clones annotation with given name from dataset to the given annotation file path.

        :param item_name: Item Name.
        :type item_name: str
        :param ann_path: Path to the annotation file.
        :type ann_path: str
        :return: None
        :rtype: :class:`NoneType`
        :raises: :class:`Exception` if ann_path is not str
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            ann_out_path = "/home/admin/work/supervisely/projects/kiwi_annotated/ds1/ann/IMG_4455.jpeg.json"
            ds.set_ann_file("IMG_1812.jpeg", ann_out_path)
        """
        if type(ann_path) is not str:
            raise TypeError(
                "Annotation path should be a string, not a {}".format(type(ann_path))
            )
        dst_ann_path = self.get_ann_path(item_name)
        copy_file(ann_path, dst_ann_path)

    def set_ann_dict(self, item_name: str, ann: Dict) -> None:
        """
        Saves given annotation with given name to dataset annotations dir in json format.

        :param item_name: Item name.
        :type item_name: str
        :param ann: Annotation as a dict in json format.
        :type ann: dict
        :return: None
        :rtype: :class:`NoneType`
        :raises: :class:`Exception` if ann_path is not str
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            ann_json = {
                "description":"",
                "size":{
                    "height":500,
                    "width":700
                },
                "tags":[],
                "objects":[],
                "customBigData":{}
            }

            ds.set_ann_dict("IMG_8888.jpeg", ann_json)
        """
        if type(ann) is not dict:
            raise TypeError("Ann should be a dict, not a {}".format(type(ann)))
        dst_ann_path = self.get_ann_path(item_name)
        dump_json_file(ann, dst_ann_path, indent=4)

    def get_item_paths(self, item_name: str) -> ItemPaths:
        """
        Generates ItemPaths object with paths to item and annotation directories for item with given name.

        :param item_name: Item name.
        :type item_name: str
        :return: ItemPaths object
        :rtype: :class:`ItemPaths<ItemPaths>`
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            img_path, ann_path = dataset.get_item_paths("IMG_0748.jpeg")
            print("img_path: " img_path, "ann_path: " ann_path)
            # Output: img_path: /home/admin/work/supervisely/projects/lemons_annotated/ds1/img/IMG_0748.jpeg
            # ann_path: /home/admin/work/supervisely/projects/lemons_annotated/ds1/ann/IMG_0748.jpeg.json
        """
        return ItemPaths(
            img_path=self.get_img_path(item_name), ann_path=self.get_ann_path(item_name)
        )

    def __len__(self):
        return len(self._item_to_ann)

    def __next__(self):
        for item_name in self._item_to_ann.keys():
            yield item_name

    def __iter__(self):
        return next(self)

    def delete_item(self, item_name: str) -> bool:
        """
        Delete image and annotation from Dataset.

        :param item_name: Item name.
        :type item_name: str
        :return: bool
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            from supervisely.project.project import Dataset
            dataset_path = "/home/admin/work/supervisely/projects/lemons_annotated"
            ds = sly.project.project.Dataset(dataset_path, sly.OpenMode.READ)

            result = dataset.delete_item("IMG_0748.jpeg")
            # Output: True
        """
        if self.item_exists(item_name):
            data_path, ann_path = self.get_item_paths(item_name)
            img_info_path = self.get_img_info_path(item_name)
            silent_remove(data_path)
            silent_remove(ann_path)
            silent_remove(img_info_path)
            self._item_to_ann.pop(item_name)
            return True
        return False

    @staticmethod
    def get_url(project_id: int, dataset_id: int) -> str:
        res = f"/projects/{project_id}/datasets/{dataset_id}/entities"
        if is_development():
            res = abs_url(res)
        return res


class Project:
    """
    Project is a parent directory for dataset. :class:`Project<Project>` object is immutable.

    :param directory: Path to project directory.
    :type directory: str
    :param mode: Determines working mode for the given project.
    :type mode: OpenMode
    :Usage example:

     .. code-block:: python

         from supervisely.project.project import Project, OpenMode
         project_path = "/home/admin/work/supervisely/projects/lemons_annotated"
         project = Project(project_path, OpenMode.READ)
    """

    dataset_class = Dataset

    class DatasetDict(KeyIndexedCollection):
        item_type = Dataset

    def __init__(self, directory: str, mode: OpenMode):
        """
        :param directory: path to the directory where the project will be saved or where it will be loaded from
        :param mode: OpenMode class object which determines in what mode to work with the project (generate exception error if not so)
        """
        if type(mode) is not OpenMode:
            raise TypeError(
                "Argument 'mode' has type {!r}. Correct type is OpenMode".format(
                    type(mode)
                )
            )

        parent_dir, name = Project._parse_path(directory)
        self._parent_dir = parent_dir
        self._name = name
        self._datasets = Project.DatasetDict()  # ds_name -> dataset object
        self._meta = None

        if mode is OpenMode.READ:
            self._read()
        else:
            self._create()

    @staticmethod
    def get_url(id: int) -> str:
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

             project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
             print(project.parent_dir)
             # Output: '/home/admin/work/supervisely/projects'
        """
        return self._parent_dir

    @property
    def name(self) -> str:
        """
        Project name.

        :return: Name
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

             project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
             print(project.name)
             # Output: 'lemons_annotated'
        """
        return self._name

    @property
    def datasets(self) -> Project.DatasetDict:
        """
        Project datasets.

        :return: Datasets
        :rtype: :class:`DatasetDict<supervisely.project.project.Project.DatasetDict>`
        :Usage example:

         .. code-block:: python

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

        :return: Project meta
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

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

             project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
             print(project.directory)
             # Output: '/home/admin/work/supervisely/projects/lemons_annotated'
        """
        return os.path.join(self.parent_dir, self.name)

    @property
    def total_items(self) -> int:
        """
        Total number of items in project.

        :return: total number of items in project
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

             project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
             print(project.total_items)
             # Output: 12
        """
        return sum(len(ds) for ds in self._datasets)

    def _get_project_meta_path(self):
        """
        :return: str (path to project meta file(meta.json))
        """
        return os.path.join(self.directory, "meta.json")

    def _read(self):
        """
        Download project from given project directory. Checks item and annotation directoris existing and dataset not empty.
        Consistency checks. Every image must have an annotation, and the correspondence must be one to one.
        """
        meta_json = load_json_file(self._get_project_meta_path())
        self._meta = ProjectMeta.from_json(meta_json)

        possible_datasets = get_subdirs(self.directory)
        for ds_name in possible_datasets:
            try:
                current_dataset = self.dataset_class(
                    os.path.join(self.directory, ds_name), OpenMode.READ
                )
                self._datasets = self._datasets.add(current_dataset)
            except Exception as ex:
                logger.warning(ex, exc_info=True)

        if self.total_items == 0:
            raise RuntimeError("Project is empty")

    def _create(self):
        """
        Creates a leaf directory and empty meta.json file. Generate exception error if project directory already exists and is not empty.
        """
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

    def validate(self):
        # @TODO: validation here
        pass

    def set_meta(self, new_meta: ProjectMeta) -> None:
        """
        Saves given meta to project directory in json format.

        :param new_meta: ProjectMeta object.
        :type new_meta: ProjectMeta
        :return: :class:`None`
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

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

    def create_dataset(self, ds_name: str) -> Dataset:
        """
        Creates a subdirectory with given name and all intermediate subdirectories for items and annotations in project directory, and also adds created dataset
        to the collection of all datasets in the project.

        :param ds_name: Dataset name.
        :type ds_name: str
        :return: Dataset object
        :rtype: :class:`Dataset<Dataset>`
        :Usage example:

         .. code-block:: python

             project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
             project.create_dataset("ds3")

             for dataset in project.datasets:
                 print(dataset.name)

             # Output: ds1
             #         ds2
             #         ds3
        """
        ds = self.dataset_class(os.path.join(self.directory, ds_name), OpenMode.CREATE)
        self._datasets = self._datasets.add(ds)
        return ds

    def _add_item_file_to_dataset(
        self, ds, item_name, item_paths, img_info_path, _validate_item, _use_hardlink
    ):
        """
        Add item file and annotation from given name and path to given dataset items directory. Generate exception error if item_name already exists in dataset or item name has unsupported extension
        :param ds: Dataset class object
        :param item_name: str
        :param item_paths: ItemPaths object
        :param _validate_item: bool
        :param _use_hardlink: bool
        """
        ds.add_item_file(
            item_name,
            item_path=item_paths.img_path,
            ann=item_paths.ann_path,
            img_info=img_info_path,
            _validate_item=_validate_item,
            _use_hardlink=_use_hardlink,
        )

    def copy_data(
        self,
        dst_directory: str,
        dst_name: Optional[str] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
    ) -> Project:
        """
        Makes a copy of the Project.

        :param dst_directory: Path to project parent directory.
        :type dst_directory: str
        :param dst_name: Project name.
        :type dst_name: str, optional
        :param _validate_item: Checks input files format.
        :type _validate_item: bool, optional
        :param _use_hardlink: If True creates a hardlink pointing to src named dst, otherwise don't.
        :type _use_hardlink: bool, optional
        :return: Project object
        :rtype: :class:`Project<Project>`
        :Usage example:

         .. code-block:: python

             project = sly.Project("/home/admin/work/supervisely/projects/lemons_annotated", sly.OpenMode.READ)
             project.copy_data("/home/admin/work/supervisely/projects/", "lemons_copy")
        """
        dst_name = dst_name if dst_name is not None else self.name
        new_project = Project(os.path.join(dst_directory, dst_name), OpenMode.CREATE)
        new_project.set_meta(self.meta)

        for ds in self:
            new_ds = new_project.create_dataset(ds.name)

            ds: Dataset
            for item_name in ds:
                item_paths = ds.get_item_paths(item_name)
                img_info_path = ds.get_img_info_path(item_name)

                item_paths = ItemPaths(
                    img_path=item_paths.img_path
                    if os.path.isfile(item_paths.img_path)
                    else None,
                    ann_path=item_paths.ann_path
                    if os.path.isfile(item_paths.ann_path)
                    else None,
                )
                img_info_path = img_info_path if os.path.isfile(img_info_path) else None

                self._add_item_file_to_dataset(
                    new_ds,
                    item_name,
                    item_paths,
                    img_info_path,
                    _validate_item,
                    _use_hardlink,
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
        progress_cb: Optional[Callable] = None,
        segmentation_type: Optional[str] = "semantic",
    ) -> None:

        _bg_class_name = "__bg__"
        _bg_obj_class = ObjClass(_bg_class_name, Bitmap, color=[0, 0, 0])

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
                raise ValueError(
                    f"Destination directory {dst_project_dir} is not empty"
                )

        src_project = Project(src_project_dir, OpenMode.READ)
        dst_meta = src_project.meta.clone()

        dst_meta, dst_mapping = dst_meta.to_segmentation_task(
            target_classes=target_classes
        )

        if (
            segmentation_type == "semantic"
            and dst_meta.obj_classes.get(_bg_class_name) is None
        ):
            dst_meta = dst_meta.add_obj_class(_bg_obj_class)

        if target_classes is not None:
            if segmentation_type == "semantic":
                if _bg_class_name not in target_classes:
                    target_classes.append(_bg_class_name)

            # check that all target classes are in destination project meta
            for class_name in target_classes:
                if dst_meta.obj_classes.get(class_name) is None:
                    raise KeyError(
                        f"Class {class_name} not found in destination project meta"
                    )

            dst_meta = dst_meta.clone(
                obj_classes=ObjClassCollection(
                    [
                        dst_meta.obj_classes.get(class_name)
                        for class_name in target_classes
                    ]
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

                seg_ann = ann.to_nonoverlapping_masks(
                    dst_mapping
                )  # rendered instances and filter classes

                if segmentation_type == "semantic":
                    seg_ann = seg_ann.add_bg_object(_bg_obj_class)

                    dst_mapping[_bg_obj_class] = _bg_obj_class
                    seg_ann = seg_ann.to_nonoverlapping_masks(
                        dst_mapping
                    )  # get_labels with bg

                    seg_ann = seg_ann.to_segmentation_task()
                elif segmentation_type == "instance":
                    pass
                elif segmentation_type == "panoptic":
                    raise NotImplementedError

                seg_path = None
                if inplace is False:
                    dst_dataset.add_item_file(item_name, img_path, seg_ann)
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
    ) -> None:
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
                raise ValueError(
                    f"Destination directory {dst_project_dir} is not empty"
                )

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

        if inplace is True:
            src_project.set_meta(det_meta)

    @staticmethod
    def remove_classes_except(
        project_dir: str,
        classes_to_keep: Optional[List[str]] = [],
        inplace: Optional[bool] = False,
    ) -> None:
        """
        Removes classes from Project with the exception of some classes.

        :param project_dir: Path to project directory.
        :type project_dir: str
        :param classes_to_keep: Classes to keep in project.
        :type classes_to_keep: List[str], optional
        :param inplace: Сheckbox that determines whether to change the source data in project or not.
        :type inplace: bool, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

             project = sly.project.project.Project(project_path, sly.OpenMode.READ)
             project.remove_classes_except(project_path, inplace=True)
        """
        classes_to_remove = []
        project = Project(project_dir, OpenMode.READ)
        for obj_class in project.meta.obj_classes:
            if obj_class.name not in classes_to_keep:
                classes_to_remove.append(obj_class.name)
        Project.remove_classes(project_dir, classes_to_remove, inplace)

    @staticmethod
    def remove_classes(
        project_dir: str,
        classes_to_remove: Optional[List[str]] = [],
        inplace: Optional[bool] = False,
    ) -> None:
        """
        Removes given classes from Project.

        :param project_dir: Path to project directory.
        :type project_dir: str
        :param classes_to_remove: Classes to remove.
        :type classes_to_remove: List[str], optional
        :param inplace: Сheckbox that determines whether to change the source data in project or not.
        :type inplace: bool, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

             project = sly.project.project.Project(project_path, sly.OpenMode.READ)
             classes_to_remove = ['lemon']
             project.remove_classes(project_path, classes_to_remove, inplace=True)
        """
        if inplace is False:
            raise ValueError(
                f"Original data will be modified. Please, set 'inplace' argument (inplace=True) directly"
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
                f"Original data will be modified. Please, set 'inplace' argument (inplace=True) directly"
            )
        if (
            without_objects is False
            and without_tags is False
            and without_objects_and_tags is False
        ):
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
    def remove_items_without_objects(
        project_dir: str, inplace: Optional[bool] = False
    ) -> None:
        """
        Remove items(images and annotations) without objects from Project.

        :param project_dir: Path to project directory.
        :type project_dir: str
        :param inplace: Сheckbox that determines whether to change the source data in project or not.
        :type inplace: bool, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            project = sly.project.project.Project(project_path, sly.OpenMode.READ)
            project.remove_items_without_objects(project_path)
        """
        Project._remove_items(
            project_dir=project_dir, without_objects=True, inplace=inplace
        )

    @staticmethod
    def remove_items_without_tags(
        project_dir: str, inplace: Optional[bool] = False
    ) -> None:
        """
        Remove items(images and annotations) without tags from Project.

        :param project_dir: Path to project directory.
        :type project_dir: str
        :param inplace: Сheckbox that determines whether to change the source data in project or not.
        :type inplace: bool, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            project = sly.project.project.Project(project_path, sly.OpenMode.READ)
            project.remove_items_without_tags(project_path)
        """
        Project._remove_items(
            project_dir=project_dir, without_tags=True, inplace=inplace
        )

    @staticmethod
    def remove_items_without_both_objects_and_tags(
        project_dir: str, inplace: Optional[bool] = False
    ) -> None:
        """
        Remove items(images and annotations) without objects and tags from Project.

        :param project_dir: Path to project directory.
        :type project_dir: str
        :param inplace: Сheckbox that determines whether to change the source data in project or not.
        :type inplace: bool, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            project = sly.project.project.Project(project_path, sly.OpenMode.READ)
            project.remove_items_without_both_objects_and_tags(project_path)
        """
        Project._remove_items(
            project_dir=project_dir, without_objects_and_tags=True, inplace=inplace
        )

    def get_item_paths(self, item_name) -> ItemPaths:
        raise NotImplementedError("Method available only for dataset")

    @staticmethod
    def get_train_val_splits_by_count(
        project_dir: str, train_count: int, val_count: int
    ) -> Tuple[List[ItemInfo], List[ItemInfo]]:
        """
        Get train and val items information from project by given train and val counts.

        :param project_dir: Path to project directory.
        :type project_dir: str
        :param train_count: Number of train items.
        :type train_count: int
        :param val_count: Number of val items.
        :type val_count: int
        :raises: :class:`ValueError` if total_count != train_count + val_count
        :return: Tuple with lists of train items information and val items information
        :rtype: :class:`Tuple[List[ItemInfo], List[ItemInfo]]`
        :Usage example:

         .. code-block:: python

            project = sly.project.project.Project(project_path, sly.OpenMode.READ)
            train_count = 4
            val_count = 2
            train_items, val_items = project.get_train_val_splits_by_count(project_path, train_count, val_count)
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
        :type project_dir: str
        :param train_tag_name: Train tag name.
        :type train_tag_name: str
        :param val_tag_name: Val tag name.
        :type val_tag_name: str
        :param untagged: Actions in case of absence of train_tag_name and val_tag_name in project.
        :type untagged: str, optional
        :raises: :class:`ValueError` if untagged not in ["ignore", "train", "val"]
        :return: Tuple with lists of train items information and val items information
        :rtype: :class:`Tuple[List[ItemInfo], List[ItemInfo]]`
        :Usage example:

         .. code-block:: python

            project = sly.project.project.Project(project_path, sly.OpenMode.READ)
            train_tag_name = 'train'
            val_tag_name = 'val'
            train_items, val_items = project.get_train_val_splits_by_tag(project_path, train_tag_name, val_tag_name)
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
        :type project_dir: str
        :param train_datasets: List of train datasets names.
        :type train_datasets: List[str]
        :param val_datasets: List of val datasets names.
        :type val_datasets: List[str]
        :raises: :class:`KeyError` if dataset name not found in project
        :return: Tuple with lists of train items information and val items information
        :rtype: :class:`Tuple[List[ItemInfo], List[ItemInfo]]`
        :Usage example:

         .. code-block:: python

            project = sly.project.project.Project(project_path, sly.OpenMode.READ)
            train_datasets = ['ds1', 'ds2']
            val_datasets = ['ds3', 'ds4']
            train_items, val_items = project.get_train_val_splits_by_dataset(project_path, train_datasets, val_datasets)
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


def read_single_project(
    dir: str, project_class: Optional[Project] = Project
) -> Project:
    """
    Read project from given directory.

    :param dir: Path to project parent directory, which must contain only project folder.
    :type dir: str
    :param project_class: Project object
    :type project_class: Project
    :return: Project class object
    :rtype: :class:`Project<Project>`
    :raises: :class:`Exception` if given directory contains more than one subdirectory.
    :Usage example:

     .. code-block:: python

            proj_dir = "/home/admin/work/supervisely/source/project" # Is a Project's parent directory containing project folder!
            project = sly.read_single_project(proj_dir)
    """
    try:
        project_fs = project_class(dir, OpenMode.READ)
        return project_fs
    except Exception:
        pass

    projects_in_dir = get_subdirs(dir)
    if len(projects_in_dir) != 1:
        raise RuntimeError("Found {} dirs instead of 1".format(len(projects_in_dir)))

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


def _download_project(
    api,
    project_id,
    dest_dir,
    dataset_ids=None,
    log_progress=False,
    batch_size=10,
    only_image_tags=False,
    save_image_info=False,
    save_images=True,
):
    dataset_ids = set(dataset_ids) if (dataset_ids is not None) else None
    project_fs = Project(dest_dir, OpenMode.CREATE)
    meta = ProjectMeta.from_json(api.project.get_meta(project_id))
    project_fs.set_meta(meta)

    if only_image_tags is True:
        id_to_tagmeta = meta.tag_metas.get_id_mapping()

    for dataset_info in api.dataset.get_list(project_id):
        dataset_id = dataset_info.id
        if dataset_ids is not None and dataset_id not in dataset_ids:
            continue

        dataset_fs = project_fs.create_dataset(dataset_info.name)
        images = api.image.get_list(dataset_id)

        ds_progress = None
        if log_progress:
            ds_progress = Progress(
                "Downloading dataset: {!r}".format(dataset_info.name),
                total_cnt=len(images),
            )

        for batch in batched(images, batch_size):
            image_ids = [image_info.id for image_info in batch]
            image_names = [image_info.name for image_info in batch]

            # download images in numpy format
            batch_imgs_bytes = api.image.download_bytes(dataset_id, image_ids)

            # download annotations in json format
            if only_image_tags is False:
                ann_infos = api.annotation.download_batch(dataset_id, image_ids)
                ann_jsons = [ann_info.annotation for ann_info in ann_infos]
            else:
                ann_jsons = []
                for image_info in batch:
                    tags = TagCollection.from_api_response(
                        image_info.tags, meta.tag_metas, id_to_tagmeta
                    )
                    tmp_ann = Annotation(
                        img_size=(image_info.height, image_info.width), img_tags=tags
                    )
                    ann_jsons.append(tmp_ann.to_json())

            for img_info, name, img_bytes, ann in zip(
                batch, image_names, batch_imgs_bytes, ann_jsons
            ):
                dataset_fs.add_item_raw_bytes(
                    item_name=name,
                    item_raw_bytes=img_bytes if save_images is True else None,
                    ann=ann,
                    img_info=img_info if save_image_info is True else None,
                )

            if log_progress:
                ds_progress.iters_done_report(len(batch))


def upload_project(
    dir: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: Optional[bool] = True,
    progress_cb: Optional[Callable] = None,
) -> Tuple[int, str]:
    """
    Uploads project to Supervisely from the given directory.

    :param dir: Path to project directory.
    :type dir: str
    :param api: Supervisely API address and token.
    :type api: Api
    :param workspace_id: Workspace ID, where project will be uploaded.
    :type workspace_id: int
    :param project_name: Name of the project in Supervisely.
    :type project_name: str, optional
    :param log_progress: Show uploading progress bar.
    :type log_progress: bool, optional
    :param progress_cb: Function to check progress.
    :type progress_cb: Function, optional
    :return: Project ID and name
    :rtype: :class:`int`, :class:`str`
    :Usage example:

     .. code-block:: python

            # Local folder with Project
            project_directory = "/home/admin/work/supervisely/source/project"
            project = sly.Project(project_directory, sly.OpenMode.READ)

            # Obtain server address and your api_token from environment variables
            # Edit those values if you run this notebook on your own PC
            address = os.environ['SERVER_ADDRESS']
            token = os.environ['API_TOKEN']

            # Initialize API object
            api = sly.Api(address, token)

            # Upload Project
            sly.upload_project(project_directory, api, workspace_id=45, project_name="Lemons")
    """
    project_fs = read_single_project(dir)
    if project_name is None:
        project_name = project_fs.name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(
        workspace_id, project_name, change_name_if_conflict=True
    )
    api.project.update_meta(project.id, project_fs.meta.to_json())

    for dataset_fs in project_fs.datasets:
        dataset = api.dataset.create(project.id, dataset_fs.name)

        dataset_fs: Dataset

        names, img_paths, ann_paths, img_infos = [], [], [], []
        for item_name in dataset_fs:
            img_path, ann_path = dataset_fs.get_item_paths(item_name)
            img_info_path = dataset_fs.get_img_info_path(item_name)

            names.append(item_name)
            img_paths.append(img_path)
            ann_paths.append(ann_path)

            if os.path.isfile(img_info_path) is True:
                img_infos.append(dataset_fs.get_image_info(item_name=item_name))

        img_paths = list(filter(lambda x: os.path.isfile(x), img_paths))
        ann_paths = list(filter(lambda x: os.path.isfile(x), ann_paths))

        if log_progress and progress_cb is None:
            ds_progress = Progress(
                "Uploading images to dataset {!r}".format(dataset.name),
                total_cnt=len(names),
            )
            progress_cb = ds_progress.iters_done_report

        if len(img_paths) != 0:
            uploaded_img_infos = api.image.upload_paths(
                dataset.id, names, img_paths, progress_cb
            )
        elif len(img_paths) == 0 and len(img_infos) != 0:
            hashes = [img_info.hash for img_info in img_infos]
            uploaded_img_infos = api.image.upload_hashes(
                dataset.id, names, hashes, progress_cb
            )
        else:
            raise ValueError(
                "Cannot upload Project: img_paths is empty and img_infos_paths is empty"
            )

        image_ids = [img_info.id for img_info in uploaded_img_infos]

        if log_progress and progress_cb is None:
            ds_progress = Progress(
                "Uploading annotations to dataset {!r}".format(dataset.name),
                total_cnt=len(img_paths),
            )
            progress_cb = ds_progress.iters_done_report
        api.annotation.upload_paths(image_ids, ann_paths, progress_cb)

    return project.id, project.name


def download_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: Optional[bool] = False,
    batch_size: Optional[int] = 10,
    cache: Optional[FileCache] = None,
    progress_cb: Optional[Callable] = None,
    only_image_tags: Optional[bool] = False,
    save_image_info: Optional[bool] = False,
    save_images: bool = True,
) -> None:
    """
    Download project from Supervisely to the given directory.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Supervisely downloadable project ID.
    :type project_id: int
    :param dest_dir: Destination directory.
    :type dest_dir: str
    :param dataset_ids: Dataset IDs.
    :type dataset_ids: List[int], optional
    :param log_progress: Show uploading progress bar.
    :type log_progress: bool, optional
    :param batch_size: The number of images in the batch when they are loaded to a host.
    :type batch_size: int, optional
    :param cache: FileCache object.
    :type cache: FileCache, optional
    :param progress_cb: Function to check progress.
    :type progress_cb: Function, optional
    :param only_image_tags: Download project with only images tags.
    :type only_image_tags: bool, optional
    :param save_image_info: Save images infos.
    :type save_image_info: bool, optional
    :return: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

            # Local destination Project folder
            save_directory = "/home/admin/work/supervisely/source/project"

            # Obtain server address and your api_token from environment variables
            # Edit those values if you run this notebook on your own PC
            address = os.environ['SERVER_ADDRESS']
            token = os.environ['API_TOKEN']

            # Initialize API object
            api = sly.Api(address, token)
            project_id = 8888

            # Upload Project
            sly.download_project(api, project_id, save_directory)
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
):
    project_info = api.project.get_info_by_id(project_id)
    project_id = project_info.id
    logger.info(
        f"Annotations are not cached (always download latest version from server)"
    )
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
            _download_dataset(
                api,
                dataset,
                dataset_id,
                cache=cache,
                progress_cb=progress_cb,
                project_meta=meta,
                only_image_tags=only_image_tags,
                save_image_info=save_image_info,
                save_images=save_images,
            )


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
    dataset,
    dataset_id,
    cache=None,
    progress_cb=None,
    project_meta: ProjectMeta = None,
    only_image_tags=False,
    save_image_info=False,
    save_images=True,
):
    images = api.image.get_list(dataset_id)
    images_to_download = images
    if only_image_tags is True:
        if project_meta is None:
            raise ValueError("Project Meta is not defined")
        id_to_tagmeta = project_meta.tag_metas.get_id_mapping()

    # copy images from cache to task folder and download corresponding annotations
    if cache:
        (
            images_to_download,
            images_in_cache,
            images_cache_paths,
        ) = _split_images_by_cache(images, cache)
        if len(images_to_download) + len(images_in_cache) != len(images):
            raise RuntimeError(
                "Error with images cache during download. Please contact support."
            )
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
                ann_info_list = api.annotation.download_batch(
                    dataset_id, img_cache_ids, progress_cb
                )
                img_name_to_ann = {
                    ann.image_id: ann.annotation for ann in ann_info_list
                }
            else:
                img_name_to_ann = {}
                for image_info in images_in_cache:
                    tags = TagCollection.from_api_response(
                        image_info.tags, project_meta.tag_metas, id_to_tagmeta
                    )
                    tmp_ann = Annotation(
                        img_size=(image_info.height, image_info.width), img_tags=tags
                    )
                    img_name_to_ann[image_info.id] = tmp_ann.to_json()
                if progress_cb is not None:
                    progress_cb(len(images_in_cache))

            for batch in batched(
                list(zip(images_in_cache, images_cache_paths)), batch_size=50
            ):
                for img_info, img_cache_path in batch:
                    item_name = _maybe_append_image_extension(
                        img_info.name, img_info.ext
                    )
                    img_info_to_add = None
                    if save_image_info is True:
                        img_info_to_add = img_info
                    dataset.add_item_file(
                        item_name,
                        item_path=img_cache_path if save_images is True else None,
                        ann=img_name_to_ann[img_info.id],
                        _validate_item=False,
                        _use_hardlink=True,
                        img_info=img_info_to_add,
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
                    dataset.img_dir,
                    _maybe_append_image_extension(img_info.name, img_info.ext),
                )
            )

        # download annotations
        if only_image_tags is False:
            ann_info_list = api.annotation.download_batch(
                dataset_id, img_ids, progress_cb
            )
            img_name_to_ann = {ann.image_id: ann.annotation for ann in ann_info_list}
        else:
            img_name_to_ann = {}
            for image_info in images_to_download:
                tags = TagCollection.from_api_response(
                    image_info.tags, project_meta.tag_metas, id_to_tagmeta
                )
                tmp_ann = Annotation(
                    img_size=(image_info.height, image_info.width), img_tags=tags
                )
                img_name_to_ann[image_info.id] = tmp_ann.to_json()
            if progress_cb is not None:
                progress_cb(len(images_to_download))

        # download images and write to dataset
        for img_info_batch in batched(images_to_download):
            images_ids_batch = [image_info.id for image_info in img_info_batch]
            images_nps = api.image.download_nps(
                dataset_id, images_ids_batch, progress_cb=progress_cb
            )

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
