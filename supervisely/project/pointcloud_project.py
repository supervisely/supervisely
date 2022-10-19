# coding: utf-8

# dict
from typing import List, Optional, Dict, Tuple, Callable, NamedTuple, Union

from supervisely.api.api import Api
from collections import namedtuple
import os
import random
import numpy as np
import shutil

from supervisely.io.fs import (
    file_exists,
    touch,
    dir_exists,
    list_files,
    get_file_name_with_ext,
    get_file_name,
    copy_file,
    silent_remove,
    remove_dir,
    ensure_base_path,
)
import supervisely.imaging.image as sly_image
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress
from supervisely._utils import batched
from supervisely.video_annotation.key_id_map import KeyIdMap

from supervisely.api.module_api import ApiField
from supervisely.collection.key_indexed_collection import KeyIndexedCollection

from supervisely.project.project import OpenMode
from supervisely.project.project import read_single_project as read_project_wrapper


from supervisely.pointcloud_annotation.pointcloud_annotation import PointcloudAnnotation
import supervisely.pointcloud as sly_pointcloud
from supervisely.project.video_project import VideoDataset, VideoProject
from supervisely.io.json import dump_json_file
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger

PointcloudItemPaths = namedtuple(
    "PointcloudItemPaths", ["pointcloud_path", "related_images_dir", "ann_path"]
)
PointcloudItemInfo = namedtuple(
    "PointcloudItemInfo",
    ["dataset_name", "name", "pointcloud_path", "related_images_dir", "ann_path"],
)


class PointcloudDataset(VideoDataset):
    item_dir_name = "pointcloud"
    related_images_dir_name = "related_images"
    annotation_class = PointcloudAnnotation

    @property
    def img_dir(self) -> str:
        logger.warn(
            f"Use 'pointcloud_dir' or 'item_dir' properties instead of 'img_dir' property of {type(self).__name__} object."
        )
        return super().img_dir

    @property
    def pointcloud_dir(self) -> str:
        return super().img_dir

    @property
    def item_dir(self) -> str:
        return super().img_dir

    @property
    def seg_dir(self) -> None:
        raise NotImplementedError(
            f"Property 'seg_dir' is not supported for {type(self).__name__} object now."
        )

    @property
    def img_info_dir(self) -> str:
        logger.warn(
            f"Use 'pointcloud_info_dir' property instead of 'img_info_dir' property of {type(self).__name__} object."
        )
        return self.pointcloud_info_dir

    @property
    def pointcloud_info_dir(self) -> str:
        return os.path.join(self.directory, "pointcloud_info")

    @staticmethod
    def _has_valid_ext(path: str) -> bool:
        return sly_pointcloud.has_valid_ext(path)

    def _get_empty_annotaion(self, item_name):
        return self.annotation_class()

    def get_img_path(self, item_name: str) -> str:
        logger.warn(
            f"Use 'get_pointcloud_path()' method instead of 'get_img_path()' method of {type(self).__name__} object."
        )
        return super().get_img_path(item_name)

    def get_img_info_path(self, item_name: str) -> None:
        raise NotImplementedError(f"Use 'get_pointcloud_info_path()' method instead of 'get_img_info_path()' method of {type(self).__name__} object.")

    def get_pointcloud_path(self, item_name: str) -> str:
        """
        Path to the given pointcloud.

        :param item_name: Pointcloud name
        :type item_name: str
        :return: Path to the given pointcloud
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            from supervisely.project.pointcloud_project import PointcloudDataset
            dataset_path = "/home/admin/work/supervisely/projects/ptc_project/ds0"
            ds = PointcloudDataset(dataset_path, sly.OpenMode.READ)

            ds.get_pointcloud_path("PTC_0748")
            # Output: RuntimeError: Item IMG_0748 not found in the project.

            ds.get_pointcloud_path("PTC_0748.pcd")
            # Output: '/home/admin/work/supervisely/projects/ptc_project/ds0/pointcloud/PTC_0748.pcd'
        """
        return super().get_img_path(item_name)

    def get_pointcloud_info(self, item_name: str) -> NamedTuple:
        """
        Information for Pointcloud with given name.

        :param item_name: Pointcloud name.
        :type item_name: str
        :return: Pointcloud with information for the given Dataset
        :rtype: :class:`NamedTuple`
        :Usage example:

         .. code-block:: python

            from supervisely.project.pointcloud_project import PointcloudDataset
            dataset_path = "/home/admin/work/supervisely/projects/ptc_project/ds0"
            ds = PointcloudDataset(dataset_path, sly.OpenMode.READ)

            info = ds.get_pointcloud_info("IMG_0748.pcd")
        """
        img_info_path = self.get_pointcloud_info_path(item_name)
        image_info_dict = load_json_file(img_info_path)
        PointcloudInfo = namedtuple("PointcloudInfo", image_info_dict)
        info = PointcloudInfo(**image_info_dict)
        return info

    def get_ann_path(self, item_name: str) -> str:
        """
        Path to the given annotation.

        :param item_name: PointcloudAnnotation name.
        :type item_name: str
        :return: Path to the given annotation
        :rtype: :class:`str`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            from supervisely.project.pointcloud_project import PointcloudDataset
            dataset_path = "/home/admin/work/supervisely/projects/ptc_project/ds0"
            ds = PointcloudDataset(dataset_path, sly.OpenMode.READ)

            ds.get_ann_path("PTC_0748")
            # Output: RuntimeError: Item PTC_0748 not found in the project.

            ds.get_ann_path("PTC_0748.pcd")
            # Output: '/home/admin/work/supervisely/projects/ptc_project/ds0/ann/IMG_0748.pcd.json'
        """
        return super().get_ann_path(item_name)

    def delete_item(self, item_name: str) -> bool:
        """
        Delete image, annotation and related images from PointcloudDataset.

        :param item_name: Item name.
        :type item_name: str
        :return: bool
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            from supervisely.project.pointcloud_project import PointcloudDataset
            dataset_path = "/home/admin/work/supervisely/projects/ptc_project/ds0"
            ds = PointcloudDataset(dataset_path, sly.OpenMode.READ)

            result = dataset.delete_item("PTC_0748.pcd")
            # Output: True
        """
        if self.item_exists(item_name):
            # TODO: what about key_id_map?
            data_path, rel_images_dir, ann_path = self.get_item_paths(item_name)
            img_info_path = self.get_pointcloud_info_path(item_name)
            silent_remove(data_path)
            silent_remove(ann_path)
            silent_remove(img_info_path)
            remove_dir(rel_images_dir)
            self._item_to_ann.pop(item_name)
            return True
        return False

    def add_item_file(
        self,
        item_name: str,
        item_path: Optional[str] = None,
        ann: Optional[Union[PointcloudAnnotation, str]] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
        item_info: Optional[NamedTuple] = None,
    ) -> None:
        """
        Adds given item file to dataset items directory, and adds given annotation to dataset ann
        directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: str
        :param item_path: Path to the item.
        :type item_path: str, optional
        :param ann: PointcloudAnnotation object or path to annotation file.
        :type ann: PointcloudAnnotation or str, optional
        :param _validate_item: Checks input files format.
        :type _validate_item: bool, optional
        :param _use_hardlink: If True creates a hardlink pointing to src named dst, otherwise don't.
        :type _use_hardlink: bool, optional
        :param item_info: NamedTuple PointcloudInfo containing information about pointcloud.
        :type item_info: NamedTuple, optional
        :return: None
        :rtype: :class:`NoneType`
        :raises: :class:`RuntimeError` if item_name already exists in dataset or item name has unsupported extension.
        :Usage example:

         .. code-block:: python

            from supervisely.project.pointcloud_project import PointcloudDataset
            dataset_path = "/home/admin/work/supervisely/projects/ptc_project/ds0"
            ds = PointcloudDataset(dataset_path, sly.OpenMode.READ)

            ann = "/home/admin/work/supervisely/projects/ptc_project/ds0/ann/PTC_777.pcd.json"
            ds.add_item_file("PTC_777.pcd", "/home/admin/work/supervisely/projects/ptc_project/ds0/pointcloud/PTC_777.pcd", ann=ann)
        """
        if item_path is None and ann is None and item_info is None:
            raise RuntimeError("No item_path or ann or item_info provided.")

        self._add_item_file(
            item_name,
            item_path,
            _validate_item=_validate_item,
            _use_hardlink=_use_hardlink,
        )
        self._add_ann_by_type(item_name, ann)

        self._add_pointcloud_info(item_name, item_info)

    def add_item_np(
        self,
        item_name: str,
        pointcloud: np.ndarray,
        ann: Optional[Union[PointcloudAnnotation, str]] = None,
        item_info: Optional[NamedTuple] = None,
    ) -> None:
        """
        Adds given numpy array as a pointcloud to dataset items directory, and adds given annotation to dataset ann directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: str
        :param pointcloud: numpy Pointcloud array [N, 3], in (X, Y, Z) format.
        :type pointcloud: np.ndarray
        :param ann: PointcloudAnnotation object or path to annotation .json file.
        :type ann: PointcloudAnnotation or str, optional
        :param item_info: NamedTuple PointcloudItemInfo containing information about Pointcloud.
        :type item_info: NamedTuple, optional
        :return: None
        :rtype: :class:`NoneType`
        :raises: :class:`Exception` if item_name already exists in dataset or item name has unsupported extension
        :Usage example:

         .. code-block:: python

            from supervisely.project.pointcloud_project import PointcloudDataset
            dataset_path = "/home/admin/work/supervisely/projects/ptc_project/ds0"
            ds = PointcloudDataset(dataset_path, sly.OpenMode.READ)

            pointcloud_path = "/home/admin/Pointclouds/ptc0.pcd"
            img_np = sly.image.read(img_path)
            ds.add_item_np("IMG_050.jpeg", img_np)
        """
        self._add_pointcloud_np(item_name, pointcloud)
        self._add_ann_by_type(item_name, ann)
        self._add_pointcloud_info(item_name, item_info)

    def _add_pointcloud_np(self, item_name, pointcloud):
        if pointcloud is None:
            return
        self._check_add_item_name(item_name)
        dst_img_path = os.path.join(self.pointcloud_dir, item_name)
        sly_pointcloud.write(dst_img_path, pointcloud)

    def _add_item_file(self, item_name, item_path, _validate_item=True, _use_hardlink=False):
        if item_path is None:
            return

        self._add_pointcloud_file(item_name, item_path, _validate_item, _use_hardlink)

    def _add_pointcloud_file(
        self, item_name, pointcloud_path, _validate_pointcloud=True, _use_hardlink=False
    ):
        """
        Add given item file to dataset items directory. Generate exception error if item_name already exists in dataset
        or item name has unsupported extension
        :param item_name: str
        :param pointcloud_path: str
        :param _validate_pointcloud: bool
        :param _use_hardlink: bool
        """

        self._check_add_item_name(item_name)
        dst_pointcloud_path = os.path.join(self.pointcloud_dir, item_name)
        if (
            pointcloud_path != dst_pointcloud_path and pointcloud_path is not None
        ):  # used only for agent + api during download project + None to optimize internal usage
            hardlink_done = False
            if _use_hardlink:
                try:
                    os.link(pointcloud_path, dst_pointcloud_path)
                    hardlink_done = True
                except OSError:
                    pass
            if not hardlink_done:
                copy_file(pointcloud_path, dst_pointcloud_path)
            if _validate_pointcloud:
                self._validate_added_item_or_die(pointcloud_path)

    @staticmethod
    def _validate_added_item_or_die(item_path):
        # Make sure we actually received a valid pointcloud file, clean it up and fail if not so.
        try:
            sly_pointcloud.validate_format(item_path)
        except (sly_pointcloud.UnsupportedPointcloudFormat, sly_pointcloud.PointcloudReadException):
            os.remove(item_path)
            raise

    def get_related_images_path(self, item_name: str) -> str:
        item_name_temp = item_name.replace(".", "_")
        rimg_dir = os.path.join(self.directory, self.related_images_dir_name, item_name_temp)
        return rimg_dir

    def get_item_paths(self, item_name: str) -> PointcloudItemPaths:
        return PointcloudItemPaths(
            pointcloud_path=self.get_pointcloud_path(item_name),
            related_images_dir=self.get_related_images_path(item_name),
            ann_path=self.get_ann_path(item_name),
        )

    def get_related_images(self, item_name: str) -> List[Tuple[str, Dict]]:
        results = []
        path = self.get_related_images_path(item_name)
        if dir_exists(path):
            files = list_files(path, sly_image.SUPPORTED_IMG_EXTS)
            for file in files:
                img_meta_path = os.path.join(path, get_file_name_with_ext(file) + ".json")
                img_meta = {}
                if file_exists(img_meta_path):
                    img_meta = load_json_file(img_meta_path)
                    if img_meta[ApiField.NAME] != get_file_name_with_ext(file):
                        raise RuntimeError("Wrong format: name field contains wrong image path")
                results.append((file, img_meta))
        return results

    def get_pointcloud_info_path(self, item_name: str) -> str:
        if item_name not in self._item_to_ann.keys():
            raise RuntimeError("Item {} not found in the project.".format(item_name))

        return os.path.join(self.pointcloud_info_dir, f"{item_name}.json")

    def _add_pointcloud_info(self, item_name, pointcloud_info=None):
        if pointcloud_info is None:
            return

        dst_info_path = self.get_pointcloud_info_path(item_name)
        ensure_base_path(dst_info_path)
        if type(pointcloud_info) is dict:
            dump_json_file(pointcloud_info, dst_info_path, indent=4)
        elif type(pointcloud_info) is str and os.path.isfile(pointcloud_info):
            shutil.copy(pointcloud_info, dst_info_path)
        else:
            # PointcloudInfo named tuple
            dump_json_file(pointcloud_info._asdict(), dst_info_path, indent=4)


class PointcloudProject(VideoProject):
    dataset_class = PointcloudDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = PointcloudDataset

    @classmethod
    def read_single(cls, dir):
        return read_project_wrapper(dir, cls)

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
        if without_objects is False and without_tags is False and without_objects_and_tags is False:
            raise ValueError(
                "One of the flags (without_objects / without_tags or without_objects_and_tags) have to be defined"
            )
        project = PointcloudProject(project_dir, OpenMode.READ)
        for dataset in project.datasets:
            items_to_delete = []
            for item_name in dataset:
                item_paths = dataset.get_item_paths(item_name)
                ann_path = item_paths.ann_path
                ann = PointcloudAnnotation.load_json_file(ann_path, project.meta)
                if (
                    (without_objects and len(ann.objects) == 0)
                    or (without_tags and len(ann.tags) == 0)
                    or (without_objects_and_tags and ann.is_empty())
                ):
                    items_to_delete.append(item_name)
            for item_name in items_to_delete:
                dataset.delete_item(item_name)

    @staticmethod
    def get_train_val_splits_by_count(
        project_dir: str, train_count: int, val_count: int
    ) -> Tuple[List[PointcloudItemInfo], List[PointcloudItemInfo]]:
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
        :rtype: :class:`Tuple[List[PointcloudItemInfo], List[PointcloudItemInfo]]`
        :Usage example:

         .. code-block:: python

            from supervisely.project.pointcloud_project import PointcloudProject
            project_path = "/home/admin/work/supervisely/projects/pointcloud_project"
            project = PointcloudProject(project_path, sly.OpenMode.READ)
            train_count = 4
            val_count = 2
            train_items, val_items = project.get_train_val_splits_by_count(project_path, train_count, val_count)
        """

        def _list_items_for_splits(project) -> List[PointcloudItemInfo]:
            items = []
            for dataset in project.datasets:
                for item_name in dataset:
                    items.append(
                        PointcloudItemInfo(
                            dataset_name=dataset.name,
                            name=item_name,
                            pointcloud_path=dataset.get_pointcloud_path(item_name),
                            related_images_dir=dataset.get_related_images_path(item_name),
                            ann_path=dataset.get_ann_path(item_name),
                        )
                    )
            return items

        project = PointcloudProject(project_dir, OpenMode.READ)
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
    ) -> Tuple[List[PointcloudItemInfo], List[PointcloudItemInfo]]:
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
        :rtype: :class:`Tuple[List[PointcloudItemInfo], List[PointcloudItemInfo]]`
        :Usage example:

         .. code-block:: python

            from supervisely.project.pointcloud_project import PointcloudProject
            project_path = "/home/admin/work/supervisely/projects/pointcloud_project"
            project = PointcloudProject(project_path, sly.OpenMode.READ)
            train_tag_name = 'train'
            val_tag_name = 'val'
            train_items, val_items = project.get_train_val_splits_by_tag(project_path, train_tag_name, val_tag_name)
        """
        untagged_actions = ["ignore", "train", "val"]
        if untagged not in untagged_actions:
            raise ValueError(
                f"Unknown untagged action {untagged}. Should be one of {untagged_actions}"
            )
        project = PointcloudProject(project_dir, OpenMode.READ)
        train_items = []
        val_items = []
        for dataset in project.datasets:
            for item_name in dataset:
                item_paths = dataset.get_item_paths(item_name)
                info = PointcloudItemInfo(
                    dataset_name=dataset.name,
                    name=item_name,
                    pointcloud_path=item_paths.pointcloud_path,
                    related_images_dir=item_paths.related_images_dir,
                    ann_path=item_paths.ann_path,
                )

                ann = PointcloudAnnotation.load_json_file(item_paths.ann_path, project.meta)
                if ann.tags.get(train_tag_name) is not None:
                    train_items.append(info)
                if ann.tags.get(val_tag_name) is not None:
                    val_items.append(info)
                if ann.tags.get(train_tag_name) is None and ann.tags.get(val_tag_name) is None:
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
    ) -> Tuple[List[PointcloudItemInfo], List[PointcloudItemInfo]]:
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
        :rtype: :class:`Tuple[List[PointcloudItemInfo], List[PointcloudItemInfo]]`
        :Usage example:

         .. code-block:: python

            from supervisely.project.pointcloud_project import PointcloudProject
            project_path = "/home/admin/work/supervisely/projects/pointcloud_project"
            project = PointcloudProject(project_path, sly.OpenMode.READ)
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
                    item_paths = dataset.get_item_paths(item_name)
                    info = PointcloudItemInfo(
                        dataset_name=dataset.name,
                        name=item_name,
                        pointcloud_path=item_paths.pointcloud_path,
                        related_images_dir=item_paths.related_images_dir,
                        ann_path=item_paths.ann_path,
                    )
                    items_list.append(info)

        project = PointcloudProject(project_dir, OpenMode.READ)
        train_items = []
        _add_items_to_list(project, train_datasets, train_items)
        val_items = []
        _add_items_to_list(project, val_datasets, val_items)
        return train_items, val_items


def download_pointcloud_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    download_items: Optional[bool] = True,
    log_progress: Optional[bool] = False,
    save_pointcloud_info: Optional[bool] = False,
    save_pointclouds: Optional[bool] = True,
    batch_size: Optional[int] = 10,
    progress_cb: Optional[Callable] = None,
) -> PointcloudProject:

    key_id_map = KeyIdMap()

    project_fs = PointcloudProject(dest_dir, OpenMode.CREATE)

    meta = ProjectMeta.from_json(api.project.get_meta(project_id))
    project_fs.set_meta(meta)

    datasets_infos = []
    if dataset_ids is not None:
        for ds_id in dataset_ids:
            datasets_infos.append(api.dataset.get_info_by_id(ds_id))
    else:
        datasets_infos = api.dataset.get_list(project_id)

    for dataset in datasets_infos:
        dataset_fs = project_fs.create_dataset(dataset.name)
        pointclouds = api.pointcloud.get_list(dataset.id)

        if log_progress:
            ds_progress = Progress(
                "Downloading dataset: {!r}".format(dataset.name), total_cnt=len(pointclouds)
            )
        for batch in batched(pointclouds, batch_size=batch_size):
            pointcloud_ids = [pointcloud_info.id for pointcloud_info in batch]
            pointcloud_names = [pointcloud_info.name for pointcloud_info in batch]

            ann_jsons = api.pointcloud.annotation.download_bulk(dataset.id, pointcloud_ids)

            for pointcloud_id, pointcloud_name, ann_json, pointcloud_info in zip(
                pointcloud_ids, pointcloud_names, ann_jsons, batch
            ):
                if pointcloud_name != ann_json[ApiField.NAME]:
                    raise RuntimeError("Error in api.video.annotation.download_batch: broken order")

                pointcloud_file_path = dataset_fs.generate_item_path(pointcloud_name)
                if download_items:
                    api.pointcloud.download_path(pointcloud_id, pointcloud_file_path)

                    related_images_path = dataset_fs.get_related_images_path(pointcloud_name)
                    related_images = api.pointcloud.get_list_related_images(pointcloud_id)
                    for rimage_info in related_images:
                        name = rimage_info[ApiField.NAME]

                        if not sly_image.has_valid_ext(name):
                            new_name = get_file_name(name)  # to fix cases like .png.json
                            if sly_image.has_valid_ext(new_name):
                                name = new_name
                                rimage_info[ApiField.NAME] = name
                            else:
                                raise RuntimeError(
                                    "Something wrong with photo context filenames.\
                                                    Please, contact support"
                                )

                        rimage_id = rimage_info[ApiField.ID]

                        path_img = os.path.join(related_images_path, name)
                        path_json = os.path.join(related_images_path, name + ".json")

                        api.pointcloud.download_related_image(rimage_id, path_img)
                        dump_json_file(rimage_info, path_json)

                else:
                    touch(pointcloud_file_path)

                dataset_fs.add_item_file(
                    pointcloud_name,
                    pointcloud_file_path if save_pointclouds else None,
                    ann=PointcloudAnnotation.from_json(ann_json, project_fs.meta, key_id_map),
                    _validate_item=False,
                    item_info=pointcloud_info if save_pointcloud_info else None,
                )
            if progress_cb is not None:
                progress_cb(len(batch))
            if log_progress:
                ds_progress.iters_done_report(len(batch))

    project_fs.set_key_id_map(key_id_map)
    return project_fs


def upload_pointcloud_project(
    directory: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: Optional[bool] = False,
) -> Tuple[int, str]:
    project_fs = PointcloudProject.read_single(directory)
    if project_name is None:
        project_name = project_fs.name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(workspace_id, project_name, ProjectType.POINT_CLOUDS)
    api.project.update_meta(project.id, project_fs.meta.to_json())

    uploaded_objects = KeyIdMap()
    for dataset_fs in project_fs:
        dataset = api.dataset.create(project.id, dataset_fs.name, change_name_if_conflict=True)

        ds_progress = None
        if log_progress:
            ds_progress = Progress(
                "Uploading dataset: {!r}".format(dataset.name), total_cnt=len(dataset_fs)
            )

        for item_name in dataset_fs:

            item_path, related_images_dir, ann_path = dataset_fs.get_item_paths(item_name)
            related_items = dataset_fs.get_related_images(item_name)

            try:
                _, meta = related_items[0]
                timestamp = meta[ApiField.META]["timestamp"]
                if timestamp:
                    item_meta = {"timestamp": timestamp}
            except (KeyError, IndexError):
                item_meta = {}

            pointcloud = api.pointcloud.upload_path(dataset.id, item_name, item_path, item_meta)

            # validate_item_annotation
            ann_json = load_json_file(ann_path)
            ann = PointcloudAnnotation.from_json(ann_json, project_fs.meta)

            # ignore existing key_id_map because the new objects will be created
            api.pointcloud.annotation.append(pointcloud.id, ann, uploaded_objects)

            # upload related_images if exist
            if len(related_items) != 0:
                rimg_infos = []
                for img_path, meta_json in related_items:
                    img = api.pointcloud.upload_related_image(img_path)[0]
                    try:
                        rimg_infos.append(
                            {
                                ApiField.ENTITY_ID: pointcloud.id,
                                ApiField.NAME: meta_json[ApiField.NAME],
                                ApiField.HASH: img,
                                ApiField.META: meta_json[ApiField.META],
                            }
                        )
                    except Exception as e:
                        logger.error(
                            "Related images uploading error.",
                            extra={
                                "pointcloud_id": pointcloud.id,
                                "meta_json": meta_json,
                                "rel_image_hash": img,
                            },
                        )
                        raise e
                try:
                    api.pointcloud.add_related_images(rimg_infos)
                except Exception as e:
                    logger.error(
                        "Related images adding error.",
                        extra={"pointcloud_id": pointcloud.id, "rel_images_info": rimg_infos},
                    )
                    raise e
            if log_progress:
                ds_progress.iters_done_report(1)

    return project.id, project_name
