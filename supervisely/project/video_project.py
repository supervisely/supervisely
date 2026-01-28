# coding: utf-8

from __future__ import annotations

import asyncio
import io
import json
import os
import tarfile
import tempfile
from collections import namedtuple
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import zstd
from tqdm import tqdm

from supervisely._utils import batched, logger
from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.module_api import ApiField
from supervisely.api.project_api import ProjectInfo
from supervisely.api.video.video_api import VideoInfo
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.io.fs import clean_dir, mkdir, touch, touch_async
from supervisely.io.json import dump_json_file, dump_json_file_async, load_json_file
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project import read_single_project as read_project_wrapper
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_settings import LabelingInterface
from supervisely.project.project_type import ProjectType
from supervisely.project.versioning.common import (
    DEFAULT_VIDEO_SCHEMA_VERSION,
    get_video_snapshot_schema,
)
from supervisely.project.versioning.schema_fields import VersionSchemaField
from supervisely.task.progress import tqdm_sly
from supervisely.video import video as sly_video
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_annotation import VideoAnnotation


class VideoItemPaths(NamedTuple):
    video_path: str
    # Full video file path of item
    ann_path: str
    # Full annotation file path of item


class VideoDataset(Dataset):
    """
    VideoDataset is where your labeled and unlabeled videos and other data files live. :class:`VideoDataset<VideoDataset>` object is immutable.

    :param directory: Path to dataset directory.
    :type directory: str
    :param mode: Determines working mode for the given dataset.
    :type mode: :class:`OpenMode<supervisely.project.project.OpenMode>`
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
        ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)
    """

    #: :class:`str`: Items data directory name
    item_dir_name = "video"

    #: :class:`str`: Annotations directory name
    ann_dir_name = "ann"

    #: :class:`str`: Items info directory name
    item_info_dir_name = "video_info"

    #: :class:`str`: Metadata directory name
    metadata_dir_name = "metadata"

    #: :class:`str`: Segmentation masks directory name
    seg_dir_name = None

    annotation_class = VideoAnnotation
    item_info_class = VideoInfo

    datasets_dir_name = "datasets"

    @property
    def project_dir(self) -> str:
        """
        Path to the video project containing the video dataset.

        :return: Path to the video project.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)
            print(ds.project_dir)
            # Output: "/home/admin/work/supervisely/projects/videos_example"
        """
        return super().project_dir

    @property
    def name(self) -> str:
        """
        Video Dataset name.

        :return: Video Dataset Name.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)
            print(ds.name)
            # Output: "ds0"
        """
        return super().name

    @property
    def directory(self) -> str:
        """
        Path to the video dataset directory.

        :return: Path to the video dataset directory.
        :rtype: :class:`str`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.directory)
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0'
        """
        return super().directory

    @property
    def metadata_directory(self) -> str:
        """
        Path to the video dataset metadata directory.

        :return: Path to the video dataset metadata directory.
        :rtype: :class:`str`
        :Usage example:
         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.metadata_directory)
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0/metadata'
        """
        return os.path.join(self.directory, self.metadata_dir_name)

    @property
    def item_dir(self) -> str:
        """
        Path to the video dataset items directory.

        :return: Path to the video dataset items directory.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.item_dir)
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0/video'
        """
        return super().item_dir

    @property
    def img_dir(self) -> str:
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Property 'img_dir' is not supported for {type(self).__name__} object."
        )

    @property
    def ann_dir(self) -> str:
        """
        Path to the video dataset annotations directory.

        :return: Path to the video dataset directory with annotations.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.ann_dir)
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0/ann'
        """
        return super().ann_dir

    @property
    def img_info_dir(self):
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Property 'img_info_dir' is not supported for {type(self).__name__} object."
        )

    @property
    def item_info_dir(self):
        """
        Path to the video dataset item with items info.

        :return: Path to the video dataset directory with items info.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.item_info_dir)
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0/video_info'
        """
        return super().item_info_dir

    @property
    def seg_dir(self):
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Property 'seg_dir' is not supported for {type(self).__name__} object."
        )

    @classmethod
    def _has_valid_ext(cls, path: str) -> bool:
        """
        Checks if file from given path is supported
        :param path: str
        :return: bool
        """
        return sly_video.has_valid_ext(path)

    def get_items_names(self) -> list:
        """
        List of video dataset item names.

        :return: List of item names.
        :rtype: :class:`list` [ :class:`str` ]
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_item_names())
            # Output: ['video_0002.mp4', 'video_0005.mp4', 'video_0008.mp4', ...]
        """
        return super().get_items_names()

    def item_exists(self, item_name: str) -> bool:
        """
        Checks if given item name belongs to the video dataset.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: True if item exist, otherwise False.
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            ds.item_exists("video_0748")     # False
            ds.item_exists("video_0748.mp4") # True
        """
        return super().item_exists(item_name)

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
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_item_path("video_0748"))
            # Output: RuntimeError: Item video_0748 not found in the project.

            print(ds.get_item_path("video_0748.mp4"))
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0/video/video_0748.mp4'
        """
        return super().get_item_path(item_name)

    def get_img_path(self, item_name: str) -> str:
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'get_img_path(item_name)' is not supported for {type(self).__name__} object."
        )

    def get_ann(
        self, item_name, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> VideoAnnotation:
        """
        Read annotation of item from json.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param project_meta: ProjectMeta object.
        :type project_meta: :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`KeyIdMap<supervisely.video_annotation.key_id_map.KeyIdMap>`, optional
        :return: VideoAnnotation object.
        :rtype: :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>`
        :raises: :class:`RuntimeError` if item not found in the project
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project_path = "/home/admin/work/supervisely/projects/videos_example"
            project = sly.VideoProject(project_path, sly.OpenMode.READ)

            ds = project.datasets.get('ds0')

            annotation = ds.get_ann("video_0748", project.meta)
            # Output: RuntimeError: Item video_0748 not found in the project.

            annotation = ds.get_ann("video_0748.mp4", project.meta)
            print(annotation.to_json())
            # Output: {
            #     "description": "",
            #     "size": {
            #         "height": 500,
            #         "width": 700
            #     },
            #     "key": "e9ef52dbbbbb490aa10f00a50e1fade6",
            #     "tags": [],
            #     "objects": [],
            #     "frames": [{
            #         "index": 0,
            #         "figures": []
            #     }]
            #     "framesCount": 1
            # }
        """
        ann_path = self.get_ann_path(item_name)
        return self.annotation_class.load_json_file(ann_path, project_meta, key_id_map)

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
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_ann_path("video_0748"))
            # Output: RuntimeError: Item video_0748 not found in the project.

            print(ds.get_ann_path("video_0748.mp4"))
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0/ann/video_0748.mp4.json'
        """
        return super().get_ann_path(item_name)

    def get_img_info_path(self, img_name: str) -> str:
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'get_img_info_path(item_name)' is not supported for {type(self).__name__} object."
        )

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
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_item_info_path("video_0748"))
            # Output: RuntimeError: Item video_0748 not found in the project.

            print(ds.get_item_info_path("video_0748.mp4"))
            # Output: '/home/admin/work/supervisely/projects/videos_example/ds0/video_info/video_0748.mp4.json'
        """
        return super().get_item_info_path(item_name)

    def get_image_info(self, item_name: str) -> None:
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'get_image_info(item_name)' is not supported for {type(self).__name__} object."
        )

    def get_item_info(self, item_name: str) -> VideoInfo:
        """
        Information for Item with given name.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: VideoInfo object.
        :rtype: :class:`VideoInfo<supervisely.api.video.video_api.VideoInfo>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            print(ds.get_item_info("video_0748.mp4"))
            # Output:
            # VideoInfo(
            #     id=198702499,
            #     name='video_0748.mp4',
            #     hash='ehYHLNFWmMNuF2fPUgnC/g/tkIIEjNIOhdbNLQXkE8Y=',
            #     team_id=16087,
            #     workspace_id=23821,
            #     project_id=124974,
            #     dataset_id=466639,
            #     path_original='/h5un6l2bnaz1vj8a9qgms4-public/videos/w/7/i4/GZYoCs...9F3kyVJ7.mp4',
            #     frames_to_timecodes=[0, 0.033367, 0.066733, 0.1001,...,10.777433, 10.8108, 10.844167],
            #     frames_count=326,
            #     frame_width=3840,
            #     frame_height=2160,
            #     created_at='2021-03-23T13:14:25.536Z',
            #     updated_at='2021-03-23T13:16:43.300Z'
            # )
        """
        item_info_path = self.get_item_info_path(item_name)
        item_info_dict = load_json_file(item_info_path)
        item_info_named_tuple = namedtuple(self.item_info_class.__name__, item_info_dict)
        return item_info_named_tuple(**item_info_dict)

    def get_seg_path(self, item_name: str) -> str:
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'get_seg_path(item_name)' is not supported for {type(self).__name__} object."
        )

    def add_item_file(
        self,
        item_name: str,
        item_path: str,
        ann: Optional[Union[VideoAnnotation, str]] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
        item_info: Optional[Union[VideoInfo, Dict, str]] = None,
    ) -> None:
        """
        Adds given item file to dataset items directory, and adds given annotation to dataset
        annotations directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param item_path: Path to the item.
        :type item_path: :class:`str`
        :param ann: VideoAnnotation object or path to annotation json file.
        :type ann: :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` or :class:`str`, optional
        :param _validate_item: Checks input files format.
        :type _validate_item: :class:`bool`, optional
        :param _use_hardlink: If True creates a hardlink pointing to src named dst, otherwise don't.
        :type _use_hardlink: :class:`bool`, optional
        :param item_info: VideoInfo object or VideoInfo object converted to dict or path to item info json file for copying to dataset item info directory.
        :type item_info: :class:`VideoInfo<supervisely.api.video.video_api.VideoInfo>` or :class:`dict` or :class:`str`, optional
        :return: None
        :rtype: NoneType
        :raises: :class:`RuntimeError` if item_name already exists in dataset or item name has unsupported extension.
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            ann = "/home/admin/work/supervisely/projects/videos_example/ds0/ann/video_8888.mp4.json"
            ds.add_item_file("video_8888.mp4", "/home/admin/work/supervisely/projects/videos_example/ds0/video/video_8888.mp4", ann=ann)
            print(ds.item_exists("video_8888.mp4"))
            # Output: True
        """
        return super().add_item_file(
            item_name=item_name,
            item_path=item_path,
            ann=ann,
            _validate_item=_validate_item,
            _use_hardlink=_use_hardlink,
            item_info=item_info,
        )

    def add_item_np(self, item_name, img, ann=None, img_info=None):
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'add_item_np()' is not supported for {type(self).__name__} object."
        )

    def add_item_raw_bytes(self, item_name, item_raw_bytes, ann=None, img_info=None):
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method 'add_item_raw_bytes()' is not supported for {type(self).__name__} object."
        )

    def get_classes_stats(
        self,
        project_meta: Optional[ProjectMeta] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        if project_meta is None:
            project = VideoProject(self.project_dir, OpenMode.READ)
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
            for ann_obj in item_ann.objects:
                class_objects[ann_obj.obj_class.name] += 1
            for video_figure in item_ann.figures:
                class_figures[video_figure.parent_object.obj_class.name] += 1
                item_class[video_figure.parent_object.obj_class.name] = True
            for obj_class in project_meta.obj_classes:
                if obj_class.name in item_class.keys():
                    class_items[obj_class.name] += 1

        result = {}
        if return_items_count:
            result["items_count"] = class_items
        if return_objects_count:
            result["objects_count"] = class_objects
        if return_figures_count:
            result["figures_count"] = class_figures
        return result

    def _get_empty_annotaion(self, item_name):
        """
        Create empty VideoAnnotation for given video
        :param item_name: str
        :return: VideoAnnotation class object
        """
        img_size, frames_count = sly_video.get_image_size_and_frames_count(item_name)
        return self.annotation_class(img_size, frames_count)

    def _add_item_raw_bytes(self, item_name, item_raw_bytes):
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method '_add_item_raw_bytes()' is not supported for {type(self).__name__} object."
        )

    def _add_img_np(self, item_name, img):
        """
        Not available for VideoDataset class object.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Method '_add_img_np()' is not supported for {type(self).__name__} object."
        )

    def _validate_added_item_or_die(self, item_path):
        """
        Make sure we actually received a valid video file, clean it up and fail if not so.
        :param item_path: str
        """
        # Make sure we actually received a valid video file, clean it up and fail if not so.
        try:
            sly_video.validate_format(item_path)
        except Exception as e:
            os.remove(item_path)
            raise e

    def set_ann(
        self, item_name: str, ann: VideoAnnotation, key_id_map: Optional[KeyIdMap] = None
    ) -> None:
        """
        Replaces given annotation for given item name to dataset annotations directory in json format.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :param ann: VideoAnnotation object.
        :type ann: :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>`
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`KeyIdMap<supervisely.video_annotation.key_id_map.KeyIdMap>`, optional
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            height, width = 500, 700
            new_ann = sly.VideoAnnotation((height, width), frames_count=0)
            ds.set_ann("video_0748.mp4", new_ann)
        """
        if type(ann) is not self.annotation_class:
            raise TypeError(
                f"Type of 'ann' should be {self.annotation_class.__name__}, not a {type(ann).__name__}"
            )
        dst_ann_path = self.get_ann_path(item_name)
        dump_json_file(ann.to_json(key_id_map), dst_ann_path, indent=4)

    def get_item_paths(self, item_name) -> VideoItemPaths:
        """
        Generates :class:`VideoItemPaths<VideoItemPaths>` object with paths to item and annotation directories for item with given name.

        :param item_name: Item name.
        :type item_name: :class:`str`
        :return: VideoItemPaths object
        :rtype: :class:`VideoItemPaths<VideoItemPaths>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            dataset_path = "/home/admin/work/supervisely/projects/videos_example/ds0"
            ds = sly.VideoDataset(dataset_path, sly.OpenMode.READ)

            video_path, ann_path = dataset.get_item_paths("video_0748.mp4")
            print("video_path:", video_path)
            print("ann_path:", ann_path)
            # Output:
            # video_path: /home/admin/work/supervisely/projects/videos_example/ds0/video/video_0748.mp4
            # ann_path: /home/admin/work/supervisely/projects/videos_example/ds0/ann/video_0748.mp4.json
        """
        return VideoItemPaths(
            video_path=self.get_item_path(item_name), ann_path=self.get_ann_path(item_name)
        )

    @staticmethod
    def get_url(project_id: int, dataset_id: int) -> str:
        """
        Get URL to dataset items list in Supervisely.

        :param project_id: :class:`VideoProject<VideoProject>` ID in Supervisely.
        :type project_id: :class:`int`
        :param dataset_id: :class:`VideoDataset<VideoDataset>` ID in Supervisely.
        :type dataset_id: :class:`int`
        :return: URL to dataset items list.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            from supervisely import VideoDataset

            project_id = 10093
            dataset_id = 45330
            ds_items_link = VideoDataset.get_url(project_id, dataset_id)

            print(ds_items_link)
            # Output: "/projects/10093/datasets/45330"
        """
        return super().get_url(project_id, dataset_id)


class VideoProject(Project):
    """
    VideoProject is a parent directory for video dataset. VideoProject object is immutable.

    :param directory: Path to video project directory.
    :type directory: :class:`str`
    :param mode: Determines working mode for the given project.
    :type mode: :class:`OpenMode<supervisely.project.project.OpenMode>`
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        project_path = "/home/admin/work/supervisely/projects/videos_example"
        project = sly.Project(project_path, sly.OpenMode.READ)
    """

    dataset_class = VideoDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = VideoDataset

    def __init__(self, directory, mode: OpenMode):
        """
        :param directory: path to the directory where the project will be saved or where it will be loaded from
        :param mode: OpenMode class object which determines in what mode to work with the project (generate exception error if not so)
        """
        self._key_id_map: KeyIdMap = None
        super().__init__(directory, mode)

    @staticmethod
    def get_url(id: int) -> str:
        """
        Get URL to video datasets list in Supervisely.

        :param id: :class:`VideoProject<VideoProject>` ID in Supervisely.
        :type id: :class:`int`
        :return: URL to datasets list.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            from supervisely import VideoProject

            project_id = 10093
            datasets_link = VideoProject.get_url(project_id)

            print(datasets_link)
            # Output: "/projects/10093/datasets"
        """
        return super().get_url(id)

    def get_classes_stats(
        self,
        dataset_names: Optional[List[str]] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        return super(VideoProject, self).get_classes_stats(
            dataset_names, return_objects_count, return_figures_count, return_items_count
        )

    def _read(self):
        """
        Download project from given project directory. Checks item and annotation directoris existing and dataset not empty.
        Consistency checks. Every video must have an annotation, and the correspondence must be one to one.
        """
        super()._read()
        self._key_id_map = KeyIdMap()
        if os.path.exists(self._get_key_id_map_path()):
            self._key_id_map = self._key_id_map.load_json(self._get_key_id_map_path())

    def _create(self):
        """
        Creates a leaf directory and empty meta.json file. Generate exception error if project directory already exists and is not empty.
        """
        super()._create()
        self.set_key_id_map(KeyIdMap())

    @property
    def key_id_map(self):
        # TODO: write docstring
        return self._key_id_map

    @property
    def type(self) -> str:
        """
        Project type.

        :return: Project type.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.VideoProject("/home/admin/work/supervisely/projects/video", sly.OpenMode.READ)
            print(project.type)
            # Output: 'videos'
        """
        return ProjectType.VIDEOS.value

    def set_key_id_map(self, new_map: KeyIdMap):
        """
        Save given KeyIdMap object to project dir in json format.
        :param new_map: KeyIdMap class object
        """
        self._key_id_map = new_map
        self._key_id_map.dump_json(self._get_key_id_map_path())

    def _get_key_id_map_path(self):
        """
        :return: str (full path to key_id_map.json)
        """
        return os.path.join(self.directory, "key_id_map.json")

    def copy_data(
        self,
        dst_directory: str,
        dst_name: Optional[str] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
    ) -> VideoProject:
        """
        Makes a copy of the :class:`VideoProject<VideoProject>`.

        :param dst_directory: Path to video project parent directory.
        :type dst_directory: :class:`str`
        :param dst_name: Video Project name.
        :type dst_name: :class:`str`, optional
        :param _validate_item: Checks input files format.
        :type _validate_item: :class:`bool`, optional
        :param _use_hardlink: If True creates a hardlink pointing to src named dst, otherwise don't.
        :type _use_hardlink: :class:`bool`, optional
        :return: VideoProject object.
        :rtype: :class:`VideoProject<VideoProject>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.VideoProject("/home/admin/work/supervisely/projects/videos_example", sly.OpenMode.READ)
            print(project.total_items)
            # Output: 6

            new_project = project.copy_data("/home/admin/work/supervisely/projects/", "videos_example_copy")
            print(new_project.total_items)
            # Output: 6
        """
        dst_name = dst_name if dst_name is not None else self.name
        new_project = VideoProject(os.path.join(dst_directory, dst_name), OpenMode.CREATE)
        new_project.set_meta(self.meta)

        for ds in self:
            new_ds = new_project.create_dataset(ds.name)

            for item_name in ds:
                item_path, ann_path = ds.get_item_paths(item_name)
                item_info_path = ds.get_item_info_path(item_name)

                item_path = item_path if os.path.isfile(item_path) else None
                ann_path = ann_path if os.path.isfile(ann_path) else None
                item_info_path = item_info_path if os.path.isfile(item_info_path) else None
                try:
                    new_ds.add_item_file(
                        item_name,
                        item_path,
                        ann_path,
                        _validate_item=_validate_item,
                        _use_hardlink=_use_hardlink,
                        item_info=item_info_path,
                    )
                except Exception as e:
                    logger.info(
                        "INFO FOR DEBUGGING",
                        extra={
                            "source_project_name": self.name,
                            "dst_directory": dst_directory,
                            "ds_name": ds.name,
                            "item_name": item_name,
                            "item_path": item_path,
                            "ann_path": ann_path,
                            "item_info": item_info_path,
                        },
                    )
                    raise e
        new_project.set_key_id_map(self.key_id_map)
        return new_project

    @staticmethod
    def to_segmentation_task(
        src_project_dir: str,
        dst_project_dir: Optional[str] = None,
        inplace: Optional[bool] = False,
        target_classes: Optional[List[str]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        segmentation_type: Optional[str] = "semantic",
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'to_segmentation_task()' is not supported for VideoProject class now."
        )

    @staticmethod
    def to_detection_task(
        src_project_dir: str,
        dst_project_dir: Optional[str] = None,
        inplace: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'to_detection_task()' is not supported for VideoProject class now."
        )

    @staticmethod
    def remove_classes_except(
        project_dir: str,
        classes_to_keep: Optional[List[str]] = None,
        inplace: Optional[bool] = False,
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'remove_classes_except()' is not supported for VideoProject class now."
        )

    @staticmethod
    def remove_classes(
        project_dir: str,
        classes_to_remove: Optional[List[str]] = None,
        inplace: Optional[bool] = False,
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'remove_classes()' is not supported for VideoProject class now."
        )

    @staticmethod
    def _remove_items(
        project_dir,
        without_objects=False,
        without_tags=False,
        without_objects_and_tags=False,
        inplace=False,
    ):
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method '_remove_items()' is not supported for VideoProject class now."
        )

    @staticmethod
    def remove_items_without_objects(project_dir: str, inplace: Optional[bool] = False) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'remove_items_without_objects()' is not supported for VideoProject class now."
        )

    @staticmethod
    def remove_items_without_tags(project_dir: str, inplace: Optional[bool] = False) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'remove_items_without_tags()' is not supported for VideoProject class now."
        )

    @staticmethod
    def remove_items_without_both_objects_and_tags(
        project_dir: str, inplace: Optional[bool] = False
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'remove_items_without_both_objects_and_tags()' is not supported for VideoProject class now."
        )

    @staticmethod
    def get_train_val_splits_by_count(project_dir: str, train_count: int, val_count: int) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'get_train_val_splits_by_count()' is not supported for VideoProject class now."
        )

    @staticmethod
    def get_train_val_splits_by_tag(
        project_dir: str,
        train_tag_name: str,
        val_tag_name: str,
        untagged: Optional[str] = "ignore",
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'get_train_val_splits_by_tag()' is not supported for VideoProject class now."
        )

    @staticmethod
    def get_train_val_splits_by_dataset(
        project_dir: str, train_datasets: List[str], val_datasets: List[str]
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'get_train_val_splits_by_tag()' is not supported for VideoProject class now."
        )

    @staticmethod
    def get_train_val_splits_by_collections(
        project_dir: str,
        train_collections: List[int],
        val_collections: List[int],
        project_id: int,
        api: Api,
    ) -> None:
        """
        Not available for VideoProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'get_train_val_splits_by_collections()' is not supported for VideoProject class now."
        )

    @classmethod
    def read_single(cls, dir):
        """
        Read project from given ditectory. Generate exception error if given dir contains more than one subdirectory
        :param dir: str
        :return: VideoProject class object
        """
        return read_project_wrapper(dir, cls)

    @staticmethod
    def download(
        api: Api,
        project_id: int,
        dest_dir: str,
        dataset_ids: List[int] = None,
        download_videos: bool = True,
        save_video_info: bool = False,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        resume_download: Optional[bool] = False,
    ) -> None:
        """
        Download video project from Supervisely to the given directory.

        :param api: Supervisely Api class object.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param project_id: Project ID in Supervisely.
        :type project_id: :class:`int`
        :param dest_dir: Directory to download video project.
        :type dest_dir: :class:`str`
        :param dataset_ids: Datasets IDs in Supervisely to download.
        :type dataset_ids: :class:`list` [ :class:`int` ], optional
        :param download_videos: Download videos from Supervisely video project in dest_dir or not.
        :type download_videos: :class:`bool`, optional
        :param save_video_info: Save video infos or not.
        :type save_video_info: :class:`bool`, optional
        :param log_progress: Log download progress or not.
        :type log_progress: :class:`bool`
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: :class:`tqdm`, optional
        :return: None
        :rtype: NoneType
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            # Local destination Project folder
            save_directory = "/home/admin/work/supervisely/source/video_project"

            # Obtain server address and your api_token from environment variables
            # Edit those values if you run this notebook on your own PC
            address = os.environ['SERVER_ADDRESS']
            token = os.environ['API_TOKEN']

            # Initialize API object
            api = sly.Api(address, token)
            project_id = 8888

            # Download Video Project
            sly.VideoProject.download(api, project_id, save_directory)
            project_fs = sly.VideoProject(save_directory, sly.OpenMode.READ)
        """
        download_video_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            download_videos=download_videos,
            save_video_info=save_video_info,
            log_progress=log_progress,
            progress_cb=progress_cb,
            resume_download=resume_download,
        )

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
        Upload video project from given directory in Supervisely.

        :param dir: Directory with video project.
        :type dir: str
        :param api: Api class object.
        :type api: Api
        :param workspace_id: Workspace ID in Supervisely to upload video project.
        :type workspace_id: int
        :param project_name: Name of video project.

        :type project_name: str
        :param log_progress: Logging progress of download video project or not.
        :type log_progress: bool
        :return: New video project ID in Supervisely and project name
        :rtype: :class:`int`, :class:`str`
        :Usage example:

        .. code-block:: python

                import supervisely as sly

                # Local folder with Video Project
                project_directory = "/home/admin/work/supervisely/source/video_project"

                # Obtain server address and your api_token from environment variables
                # Edit those values if you run this notebook on your own PC
                address = os.environ['SERVER_ADDRESS']
                token = os.environ['API_TOKEN']

                # Initialize API object
                api = sly.Api(address, token)

                # Upload Video Project
                project_id, project_name = sly.VideoProject.upload(
                    project_directory,
                    api,
                    workspace_id=45,
                    project_name="My Video Project"
                )
        """
        return upload_video_project(
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
        semaphore: Optional[Union[asyncio.Semaphore, int]] = None,
        dataset_ids: List[int] = None,
        download_videos: bool = True,
        save_video_info: bool = False,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        include_custom_data: bool = False,
        resume_download: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        Download video project from Supervisely to the given directory asynchronously.

        :param api: Supervisely Api class object.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param project_id: Project ID in Supervisely.
        :type project_id: :class:`int`
        :param dest_dir: Directory to download video project.
        :type dest_dir: :class:`str`
        :param semaphore: Semaphore to limit the number of concurrent downloads of items.
        :type semaphore: :class:`asyncio.Semaphore` or :class:`int`, optional
        :param dataset_ids: Datasets IDs in Supervisely to download.
        :type dataset_ids: :class:`list` [ :class:`int` ], optional
        :param download_videos: Download videos from Supervisely video project in dest_dir or not.
        :type download_videos: :class:`bool`, optional
        :param save_video_info: Save video infos or not.
        :type save_video_info: :class:`bool`, optional
        :param log_progress: Log download progress or not.
        :type log_progress: :class:`bool`
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: :class:`tqdm`, optional
        :param include_custom_data: Include custom data in the download.
        :type include_custom_data: :class:`bool`, optional
        :return: None
        :rtype: NoneType
        :Usage example:

        .. code-block:: python

            import supervisely as sly
            from supervisely._utils import run_coroutine

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            save_directory = "/home/admin/work/supervisely/source/video_project"
            project_id = 8888

            coroutine = sly.VideoProject.download_async(api, project_id, save_directory)
            run_coroutine(coroutine)

        """
        await download_video_project_async(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            semaphore=semaphore,
            dataset_ids=dataset_ids,
            download_videos=download_videos,
            save_video_info=save_video_info,
            log_progress=log_progress,
            progress_cb=progress_cb,
            include_custom_data=include_custom_data,
            resume_download=resume_download,
            **kwargs,
        )

    # --------------------- #
    # Video Data Versioning #
    # --------------------- #
    @staticmethod
    def download_bin(
        api: Api,
        project_id: int,
        dest_dir: Optional[str] = None,
        dataset_ids: Optional[List[int]] = None,
        batch_size: int = 50,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        return_bytesio: bool = False,
    ) -> Union[str, io.BytesIO]:
        """
        Download video project snapshot in Arrow/Parquet-based binary format.

        Result is a .tar.zst archive containing:
            - project_info.json
            - project_meta.json
            - key_id_map.json
            - manifest.json
            - datasets.parquet
            - videos.parquet
            - objects.parquet
            - figures.parquet

        :param api: Supervisely API client.
        :type api: Api
        :param project_id: Source project ID.
        :type project_id: int
        :param dest_dir: Directory to save the resulting ``.tar.zst`` file. Required if ``return_bytesio`` is False.
        :type dest_dir: Optional[str]
        :param dataset_ids: Optional list of dataset IDs to include. If provided, only those datasets (and their videos/annotations) will be included in the snapshot.
        :type dataset_ids: Optional[List[int]]
        :param batch_size: Batch size for downloading video annotations.
        :type batch_size: int
        :param log_progress: If True, shows progress (uses internal tqdm progress bars) when ``progress_cb`` is not provided.
        :type log_progress: bool
        :param progress_cb: Optional progress callback. Can be a ``tqdm``-like callable or a function accepting an integer increment.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param return_bytesio: If True, return the snapshot as :class:`io.BytesIO`. If False, write the snapshot to ``dest_dir`` and return the output file path.
        :type return_bytesio: bool
        :return: Either output file path (``.tar.zst``) when ``return_bytesio`` is False, or an in-memory snapshot stream when ``return_bytesio`` is True.
        :rtype: Union[str, io.BytesIO]
        """
        if dest_dir is None and not return_bytesio:
            raise ValueError(
                "dest_dir must be specified if return_bytesio is False in VideoProject.download_bin"
            )

        snapshot_io = VideoProject.build_snapshot(
            api,
            project_id=project_id,
            dataset_ids=dataset_ids,
            batch_size=batch_size,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

        if return_bytesio:
            snapshot_io.seek(0)
            return snapshot_io

        project_info = api.project.get_info_by_id(project_id)
        os.makedirs(dest_dir, exist_ok=True)
        out_path = os.path.join(
            dest_dir,
            f"{project_info.id}_{project_info.name}.tar.zst",
        )
        with open(out_path, "wb") as dst:
            dst.write(snapshot_io.read())
        return out_path

    @staticmethod
    def upload_bin(
        api: Api,
        file: Union[str, io.BytesIO],
        workspace_id: int,
        project_name: Optional[str] = None,
        with_custom_data: bool = True,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_missed: bool = False,
    ) -> "ProjectInfo":
        """
        Restore a video project from an Arrow/Parquet-based binary snapshot.

        :param api: Supervisely API client.
        :type api: Api
        :param file: Snapshot file path (``.tar.zst``) or in-memory snapshot stream.
        :type file: Union[str, io.BytesIO]
        :param workspace_id: Target workspace ID where the project will be created.
        :type workspace_id: int
        :param project_name: Optional new project name. If not provided, the name from the snapshot will be used. If the name already exists in the workspace, a free name will be chosen.
        :type project_name: Optional[str]
        :param with_custom_data: If True, restore project/dataset/video custom data (when present in the snapshot).
        :type with_custom_data: bool
        :param log_progress: If True, shows progress (uses internal tqdm progress bars) when ``progress_cb`` is not provided.
        :type log_progress: bool
        :param progress_cb: Optional progress callback. Can be a ``tqdm``-like callable or a function accepting an integer increment.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param skip_missed: If True, skip videos that are missing on server when restoring by hash.
        :type skip_missed: bool
        :return: Info of the newly created project.
        :rtype: ProjectInfo
        """
        if isinstance(file, io.BytesIO):
            snapshot_bytes = file.getvalue()
        else:
            with open(file, "rb") as f:
                snapshot_bytes = f.read()

        return VideoProject.restore_snapshot(
            api,
            snapshot_bytes=snapshot_bytes,
            workspace_id=workspace_id,
            project_name=project_name,
            with_custom_data=with_custom_data,
            log_progress=log_progress,
            progress_cb=progress_cb,
            skip_missed=skip_missed,
        )

    @staticmethod
    def build_snapshot(
        api: Api,
        project_id: int,
        dataset_ids: Optional[List[int]] = None,
        batch_size: int = 50,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        schema_version: str = DEFAULT_VIDEO_SCHEMA_VERSION,
    ) -> io.BytesIO:
        """
        Create a video project snapshot in Arrow/Parquet+tar.zst format and return it as BytesIO.
        """
        try:
            import pyarrow  # pylint: disable=import-error
            import pyarrow.parquet as parquet  # pylint: disable=import-error
        except Exception as e:
            raise RuntimeError(
                "pyarrow is required to build video snapshot. Please install pyarrow."
            ) from e

        project_info = api.project.get_info_by_id(project_id)
        meta = ProjectMeta.from_json(api.project.get_meta(project_id, with_settings=True))
        key_id_map = KeyIdMap()
        snapshot_schema = get_video_snapshot_schema(schema_version)

        tmp_root = tempfile.mkdtemp()
        payload_dir = os.path.join(tmp_root, "payload")
        mkdir(payload_dir)

        try:
            # project_info / meta
            proj_info_path = os.path.join(payload_dir, "project_info.json")
            dump_json_file(project_info._asdict(), proj_info_path)

            proj_meta_path = os.path.join(payload_dir, "project_meta.json")
            dump_json_file(meta.to_json(), proj_meta_path)

            datasets_rows: List[dict] = []
            videos_rows: List[dict] = []
            objects_rows: List[dict] = []
            figures_rows: List[dict] = []

            dataset_ids_filter = set(dataset_ids) if dataset_ids is not None else None

            # api.dataset.tree() doesn't include custom_data
            ds_custom_data_by_id: Dict[int, dict] = {}
            try:
                for ds in api.dataset.get_list(
                    project_id, recursive=True, include_custom_data=True
                ):
                    if getattr(ds, "custom_data", None) is not None:
                        ds_custom_data_by_id[ds.id] = ds.custom_data
            except Exception:
                ds_custom_data_by_id = {}

            for parents, ds_info in api.dataset.tree(project_id):
                if dataset_ids_filter is not None and ds_info.id not in dataset_ids_filter:
                    continue

                full_path = Dataset._get_dataset_path(ds_info.name, parents)
                ds_custom_data = ds_custom_data_by_id.get(ds_info.id)
                datasets_rows.append(
                    snapshot_schema.dataset_row_from_ds_info(
                        ds_info, full_path=full_path, custom_data=ds_custom_data
                    )
                )

                videos = api.video.get_list(ds_info.id)
                ds_progress = progress_cb
                if log_progress and progress_cb is None:
                    ds_progress = tqdm_sly(
                        desc=f"Collecting videos from '{ds_info.name}'",
                        total=len(videos),
                    )

                for batch in batched(videos, batch_size):
                    video_ids = [v.id for v in batch]
                    ann_jsons = api.video.annotation.download_bulk(ds_info.id, video_ids)

                    for video_info, ann_json in zip(batch, ann_jsons):
                        if video_info.name != ann_json[ApiField.VIDEO_NAME]:
                            raise RuntimeError(
                                "Error in api.video.annotation.download_bulk: broken order"
                            )

                        videos_rows.append(
                            snapshot_schema.video_row_from_video_info(
                                video_info, src_dataset_id=ds_info.id, ann_json=ann_json
                            )
                        )

                        video_ann = VideoAnnotation.from_json(ann_json, meta, key_id_map)
                        obj_key_to_src_id: Dict[str, int] = {}
                        for obj in video_ann.objects:
                            src_obj_id = len(objects_rows) + 1
                            obj_key_to_src_id[obj.key().hex] = src_obj_id
                            objects_rows.append(
                                snapshot_schema.object_row_from_object(
                                    obj, src_object_id=src_obj_id, src_video_id=video_info.id
                                )
                            )

                        for frame in video_ann.frames:
                            for fig in frame.figures:
                                parent_key = fig.parent_object.key().hex
                                src_obj_id = obj_key_to_src_id.get(parent_key)
                                if src_obj_id is None:
                                    logger.warning(
                                        f"Figure parent object with key '{parent_key}' "
                                        f"not found in objects for video '{video_info.name}'"
                                    )
                                    continue
                                figures_rows.append(
                                    snapshot_schema.figure_row_from_figure(
                                        fig,
                                        figure_row_idx=len(figures_rows),
                                        src_object_id=src_obj_id,
                                        src_video_id=video_info.id,
                                        frame_index=frame.index,
                                    )
                                )

                    if ds_progress is not None:
                        ds_progress(len(batch))

            # key_id_map.json
            key_id_map_path = os.path.join(payload_dir, "key_id_map.json")
            key_id_map.dump_json(key_id_map_path)

            # Arrow schemas
            tables_meta = []
            datasets_schema = snapshot_schema.datasets_schema(pyarrow)
            videos_schema = snapshot_schema.videos_schema(pyarrow)
            objects_schema = snapshot_schema.objects_schema(pyarrow)
            figures_schema = snapshot_schema.figures_schema(pyarrow)

            if datasets_rows:
                ds_table = pyarrow.Table.from_pylist(datasets_rows, schema=datasets_schema)
                ds_path = os.path.join(payload_dir, "datasets.parquet")
                parquet.write_table(ds_table, ds_path)
                tables_meta.append(
                    {
                        "name": "datasets",
                        "path": "datasets.parquet",
                        "row_count": ds_table.num_rows,
                    }
                )

            if videos_rows:
                v_table = pyarrow.Table.from_pylist(videos_rows, schema=videos_schema)
                v_path = os.path.join(payload_dir, "videos.parquet")
                parquet.write_table(v_table, v_path)
                tables_meta.append(
                    {
                        "name": "videos",
                        "path": "videos.parquet",
                        "row_count": v_table.num_rows,
                    }
                )

            if objects_rows:
                o_table = pyarrow.Table.from_pylist(objects_rows, schema=objects_schema)
                o_path = os.path.join(payload_dir, "objects.parquet")
                parquet.write_table(o_table, o_path)
                tables_meta.append(
                    {
                        "name": "objects",
                        "path": "objects.parquet",
                        "row_count": o_table.num_rows,
                    }
                )

            if figures_rows:
                f_table = pyarrow.Table.from_pylist(figures_rows, schema=figures_schema)
                f_path = os.path.join(payload_dir, "figures.parquet")
                parquet.write_table(f_table, f_path)
                tables_meta.append(
                    {
                        "name": "figures",
                        "path": "figures.parquet",
                        "row_count": f_table.num_rows,
                    }
                )

            manifest = {
                VersionSchemaField.SCHEMA_VERSION: schema_version,
                VersionSchemaField.TABLES: tables_meta,
            }
            manifest_path = os.path.join(payload_dir, "manifest.json")
            dump_json_file(manifest, manifest_path)

            tar_path = os.path.join(tmp_root, "snapshot.tar")
            with tarfile.open(tar_path, "w") as tar:
                tar.add(payload_dir, arcname=".")

            chunk_size = 1024 * 1024 * 50  # 50 MiB
            zst_path = os.path.join(tmp_root, "snapshot.tar.zst")
            # Try streaming compression first, fallback to single-shot
            try:
                cctx = zstd.ZstdCompressor()
                with open(tar_path, "rb") as src, open(zst_path, "wb") as dst:
                    try:
                        stream = cctx.stream_writer(dst, closefd=False)
                    except TypeError:
                        stream = cctx.stream_writer(dst)
                    with stream as compressor:
                        while True:
                            chunk = src.read(chunk_size)
                            if not chunk:
                                break
                            compressor.write(chunk)
            # Fallback: single-shot compression
            except Exception:
                with open(tar_path, "rb") as src, open(zst_path, "wb") as dst:
                    dst.write(zstd.compress(src.read()))

            with open(zst_path, "rb") as f:
                outio = io.BytesIO(f.read())
            outio.seek(0)
            return outio

        finally:
            try:
                clean_dir(tmp_root)
            except Exception:
                pass

    @staticmethod
    def restore_snapshot(
        api: Api,
        snapshot_bytes: bytes,
        workspace_id: int,
        project_name: Optional[str] = None,
        with_custom_data: bool = True,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_missed: bool = False,
    ) -> ProjectInfo:
        """
        Restore a video project from a snapshot and return ProjectInfo.
        """
        try:
            import pyarrow  # pylint: disable=import-error
            import pyarrow.parquet as parquet  # pylint: disable=import-error
        except Exception as e:
            raise RuntimeError(
                "pyarrow is required to restore video snapshot. Please install pyarrow."
            ) from e

        tmp_root = tempfile.mkdtemp()
        payload_dir = os.path.join(tmp_root, "payload")
        mkdir(payload_dir)

        try:
            try:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(io.BytesIO(snapshot_bytes)) as reader:
                    with tarfile.open(fileobj=reader, mode="r|") as tar:
                        tar.extractall(payload_dir)
            except Exception:
                tar_bytes = zstd.decompress(snapshot_bytes)
                with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
                    tar.extractall(payload_dir)

            proj_info_path = os.path.join(payload_dir, "project_info.json")
            proj_meta_path = os.path.join(payload_dir, "project_meta.json")
            key_id_map_path = os.path.join(payload_dir, "key_id_map.json")
            manifest_path = os.path.join(payload_dir, "manifest.json")

            project_info_json = load_json_file(proj_info_path)
            meta_json = load_json_file(proj_meta_path)
            manifest = load_json_file(manifest_path)

            meta = ProjectMeta.from_json(meta_json)
            _ = KeyIdMap().load_json(key_id_map_path)

            schema_version = manifest.get(VersionSchemaField.SCHEMA_VERSION) or manifest.get(
                "schema_version"
            )
            try:
                _ = get_video_snapshot_schema(schema_version)
            except Exception:
                raise RuntimeError(f"Unsupported video snapshot schema_version: {schema_version}")

            src_project_name = project_info_json.get("name")
            src_project_desc = project_info_json.get("description")
            src_project_readme = project_info_json.get("readme")
            if project_name is None:
                project_name = src_project_name

            if api.project.exists(workspace_id, project_name):
                project_name = api.project.get_free_name(workspace_id, project_name)

            project = api.project.create(
                workspace_id,
                project_name,
                ProjectType.VIDEOS,
                src_project_desc,
                readme=src_project_readme,
            )
            new_meta = api.project.update_meta(project.id, meta.to_json())

            is_multiview = False
            try:
                if new_meta.labeling_interface == LabelingInterface.MULTIVIEW:
                    is_multiview = True
            except AttributeError:
                is_multiview = False

            if with_custom_data:
                src_custom_data = project_info_json.get("custom_data") or {}
                try:
                    api.project.update_custom_data(project.id, src_custom_data, silent=True)
                except Exception:
                    logger.warning("Failed to restore project custom_data from snapshot")

            if progress_cb is not None:
                log_progress = False

            # Datasets
            ds_rows = []
            datasets_path = os.path.join(payload_dir, "datasets.parquet")
            if os.path.exists(datasets_path):
                ds_table = parquet.read_table(datasets_path)
                ds_rows = ds_table.to_pylist()

                ds_rows.sort(
                    key=lambda r: (
                        r["parent_src_dataset_id"] is not None,
                        r["parent_src_dataset_id"],
                    )
                )

            dataset_mapping: Dict[int, DatasetInfo] = {}
            for row in ds_rows:
                src_ds_id = row["src_dataset_id"]
                parent_src_id = row["parent_src_dataset_id"]
                if parent_src_id is not None:
                    parent_ds = dataset_mapping.get(parent_src_id)
                    parent_id = parent_ds.id if parent_ds is not None else None
                else:
                    parent_id = None

                custom_data = None
                if with_custom_data:
                    raw_cd = row.get("custom_data")
                    if isinstance(raw_cd, str) and raw_cd.strip():
                        try:
                            custom_data = json.loads(raw_cd)
                        except Exception:
                            logger.warning(
                                f"Failed to parse dataset custom_data for '{row.get('name')}', skipping it."
                            )
                    elif isinstance(raw_cd, dict):
                        custom_data = raw_cd

                ds = api.dataset.create(
                    project.id,
                    name=row["name"],
                    description=row["description"],
                    parent_id=parent_id,
                    custom_data=custom_data,
                )
                if with_custom_data and custom_data is not None:
                    try:
                        api.dataset.update_custom_data(ds.id, custom_data)
                    except Exception:
                        logger.warning(
                            f"Failed to restore custom_data for dataset '{row.get('name')}'"
                        )
                dataset_mapping[src_ds_id] = ds

            # Videos
            v_rows = []
            videos_path = os.path.join(payload_dir, "videos.parquet")
            if os.path.exists(videos_path):
                v_table = parquet.read_table(videos_path)
                v_rows = v_table.to_pylist()

            videos_by_dataset: Dict[int, List[dict]] = {}
            for row in v_rows:
                src_ds_id = row["src_dataset_id"]
                videos_by_dataset.setdefault(src_ds_id, []).append(row)

            src_to_new_video: Dict[int, VideoInfo] = {}

            for src_ds_id, rows in videos_by_dataset.items():
                ds_info = dataset_mapping.get(src_ds_id)
                if ds_info is None:
                    logger.warning(
                        f"Dataset with src id={src_ds_id} not found in mapping. "
                        f"Skipping its videos."
                    )
                    continue

                dataset_id = ds_info.id
                hashed_rows = [r for r in rows if r.get("hash")]
                link_rows = [r for r in rows if not r.get("hash") and r.get("link")]

                ds_progress = progress_cb
                if log_progress and progress_cb is None:
                    ds_progress = tqdm_sly(
                        desc=f"Uploading videos to '{ds_info.name}'",
                        total=len(rows),
                    )

                if hashed_rows:
                    if skip_missed:
                        existing_hashes = api.video.check_existing_hashes(
                            list({r["hash"] for r in hashed_rows})
                        )
                        kept_hashed_rows = [r for r in hashed_rows if r["hash"] in existing_hashes]
                        if not kept_hashed_rows:
                            logger.warning(
                                f"All hashed videos for dataset '{ds_info.name}' "
                                f"are missing on server; nothing to upload."
                            )
                        hashed_rows = kept_hashed_rows

                    hashes = [r["hash"] for r in hashed_rows]
                    names = [r["name"] for r in hashed_rows]
                    metas: List[dict] = []
                    for r in hashed_rows:
                        meta_dict: dict = {}
                        if r.get("meta"):
                            try:
                                meta_dict.update(json.loads(r["meta"]))
                            except Exception:
                                pass
                        metas.append(meta_dict)

                    if hashes:
                        new_infos = api.video.upload_hashes(
                            dataset_id,
                            names=names,
                            hashes=hashes,
                            metas=metas,
                            progress_cb=ds_progress,
                        )
                        for row, new_info in zip(hashed_rows, new_infos):
                            src_to_new_video[row["src_video_id"]] = new_info
                            if with_custom_data and row.get("custom_data"):
                                try:
                                    cd = json.loads(row["custom_data"])
                                    api.video.update_custom_data(new_info.id, cd)
                                except Exception:
                                    logger.warning(
                                        f"Failed to restore custom_data for video '{new_info.name}'"
                                    )

                if link_rows:
                    links = [r["link"] for r in link_rows]
                    names = [r["name"] for r in link_rows]
                    metas: List[dict] = []
                    for r in link_rows:
                        meta_dict: dict = {}
                        if r.get("meta"):
                            try:
                                meta_dict.update(json.loads(r["meta"]))
                            except Exception:
                                pass
                        metas.append(meta_dict)

                    new_infos_links = api.video.upload_links(
                        dataset_id,
                        links=links,
                        names=names,
                        metas=metas,
                        progress_cb=ds_progress,
                    )
                    for row, new_info in zip(link_rows, new_infos_links):
                        src_to_new_video[row["src_video_id"]] = new_info
                        if with_custom_data and row.get("custom_data"):
                            try:
                                cd = json.loads(row["custom_data"])
                                api.video.update_custom_data(new_info.id, cd)
                            except Exception:
                                logger.warning(
                                    f"Failed to restore custom_data for video '{new_info.name}'"
                                )

                if ds_progress is not None:
                    ds_progress(len(rows))

            # Annotations
            ann_temp_dir = os.path.join(tmp_root, "anns")
            mkdir(ann_temp_dir)

            anns_by_dataset: Dict[int, List[Tuple[int, str]]] = {}
            for row in v_rows:
                src_vid = row["src_video_id"]
                new_info = src_to_new_video.get(src_vid)
                if new_info is None:
                    continue
                src_ds_id = row["src_dataset_id"]
                anns_by_dataset.setdefault(src_ds_id, []).append((new_info.id, row["ann_json"]))

            for src_ds_id, items in anns_by_dataset.items():
                ds_info = dataset_mapping.get(src_ds_id)
                if ds_info is None:
                    continue

                video_ids: List[int] = []
                ann_paths: List[str] = []

                for vid_id, ann_json_str in items:
                    video_ids.append(vid_id)
                    ann_path = os.path.join(ann_temp_dir, f"{vid_id}.json")
                    try:
                        parsed = json.loads(ann_json_str)
                    except Exception:
                        logger.warning(
                            f"Failed to parse ann_json for restored video id={vid_id}, "
                            f"skipping its annotation."
                        )
                        continue
                    dump_json_file(parsed, ann_path)
                    ann_paths.append(ann_path)

                if not video_ids:
                    continue

                anns_progress = progress_cb
                if log_progress and progress_cb is None:
                    anns_progress = tqdm_sly(
                        desc=f"Uploading annotations to '{ds_info.name}'",
                        total=len(video_ids),
                        leave=False,
                    )
                key_id_map = KeyIdMap()
                multiview_key_id_map = KeyIdMap()

                for vid_id, ann_path in zip(video_ids, ann_paths):
                    try:
                        ann_json = load_json_file(ann_path)
                        ann = VideoAnnotation.from_json(
                            ann_json,
                            new_meta,
                            key_id_map=key_id_map,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to deserialize annotation for restored video id={vid_id}: {e}"
                        )
                        continue

                    try:
                        if not is_multiview:
                            api.video.annotation.append(vid_id, ann)
                        else:
                            api.video.annotation.upload_anns_multiview(
                                [vid_id], [ann], key_id_map=multiview_key_id_map
                            )
                        if anns_progress is not None:
                            anns_progress(1)
                    except Exception as e:
                        logger.warning(
                            f"Failed to upload annotation for dataset '{ds_info.name}', "
                            f"video id={vid_id}: {e}"
                        )
                        continue

            return project

        finally:
            try:
                clean_dir(tmp_root)
            except Exception:
                pass

    # --------------------- #


def download_video_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    download_videos: Optional[bool] = True,
    save_video_info: Optional[bool] = False,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    include_custom_data: Optional[bool] = False,
    resume_download: Optional[bool] = False,
) -> None:
    """
    Download video project to the local directory.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Project ID to download
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded. Datasets could be downloaded from different projects but with the same data type.
    :type dataset_ids: list(int), optional
    :param download_videos: Include videos in the download.
    :type download_videos: bool, optional
    :param save_video_info: Include video info in the download.
    :type save_video_info: bool, optional
    :param log_progress: Show downloading logs in the output.
    :type log_progress: bool
    :param progress_cb: Function for tracking the download progress.
    :type progress_cb: tqdm or callable, optional

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

        # Download video project
        project_id = 17758
        project_info = api.project.get_info_by_id(project_id)
        num_videos = project_info.items_count

        p = tqdm(desc="Downloading video project", total=num_videos)
        sly.download(
            api,
            project_id,
            dest_dir,
            progress_cb=p,
        )
    """
    LOG_BATCH_SIZE = 1

    key_id_map = KeyIdMap()

    project_fs = None
    meta = ProjectMeta.from_json(api.project.get_meta(project_id, with_settings=True))
    if os.path.exists(dest_dir) and resume_download:
        dump_json_file(meta.to_json(), os.path.join(dest_dir, "meta.json"))
        try:
            project_fs = VideoProject(dest_dir, OpenMode.READ)
        except RuntimeError as e:
            if "Project is empty" in str(e):
                clean_dir(dest_dir)
                project_fs = None
            else:
                raise
    if project_fs is None:
        project_fs = VideoProject(dest_dir, OpenMode.CREATE)
    project_fs.set_meta(meta)

    if progress_cb is not None:
        log_progress = False

    dataset_ids = set(dataset_ids) if (dataset_ids is not None) else None
    existing_datasets = {dataset.path: dataset for dataset in project_fs.datasets}
    for parents, dataset in api.dataset.tree(project_id):
        if dataset_ids is not None and dataset.id not in dataset_ids:
            continue

        dataset_path = Dataset._get_dataset_path(dataset.name, parents)
        if dataset_path in existing_datasets:
            dataset_fs = existing_datasets[dataset_path]
        else:
            dataset_fs = project_fs.create_dataset(dataset.name, dataset_path)
        videos = api.video.get_list(dataset.id)

        ds_progress = progress_cb
        if log_progress:
            ds_progress = tqdm_sly(
                desc="Downloading videos from {!r}".format(dataset.name),
                total=len(videos),
            )

        for batch in batched(videos, batch_size=LOG_BATCH_SIZE):
            video_ids = [video_info.id for video_info in batch]
            video_names = [video_info.name for video_info in batch]
            custom_datas = [video_info.custom_data for video_info in batch]

            try:
                ann_jsons = api.video.annotation.download_bulk(dataset.id, video_ids)
            except Exception as e:
                logger.info(
                    "INFO FOR DEBUGGING",
                    extra={
                        "project_id": project_id,
                        "dataset_id": dataset.id,
                        "video_ids": video_ids,
                    },
                )
                raise e

            for video_id, video_name, custom_data, ann_json, video_info in zip(
                video_ids, video_names, custom_datas, ann_jsons, batch
            ):
                if video_name != ann_json[ApiField.VIDEO_NAME]:
                    raise RuntimeError("Error in api.video.annotation.download_batch: broken order")

                video_file_path = dataset_fs.generate_item_path(video_name)

                if include_custom_data:
                    CUSTOM_DATA_DIR = os.path.join(dest_dir, dataset.name, "custom_data")
                    mkdir(CUSTOM_DATA_DIR)
                    custom_data_path = os.path.join(CUSTOM_DATA_DIR, f"{video_name}.json")
                    dump_json_file(custom_data, custom_data_path)

                if download_videos:
                    try:
                        video_file_size = video_info.file_meta.get("size")
                        if ds_progress is not None and video_file_size is not None:
                            item_progress = tqdm_sly(
                                desc=f"Downloading '{video_name}'",
                                total=int(video_file_size),
                                unit="B",
                                unit_scale=True,
                                leave=False,
                            )
                            api.video.download_path(video_id, video_file_path, item_progress)
                        else:
                            api.video.download_path(video_id, video_file_path)
                    except Exception as e:
                        logger.info(
                            "INFO FOR DEBUGGING",
                            extra={
                                "project_id": project_id,
                                "dataset_id": dataset.id,
                                "video_id": video_id,
                                "video_file_path": video_file_path,
                            },
                        )
                        raise e
                else:
                    touch(video_file_path)
                item_info = video_info._asdict() if save_video_info else None
                try:
                    video_ann = VideoAnnotation.from_json(ann_json, project_fs.meta, key_id_map)
                except Exception as e:
                    logger.info(
                        "INFO FOR DEBUGGING",
                        extra={
                            "project_id": project_id,
                            "dataset_id": dataset.id,
                            "video_id": video_id,
                            "video_name": video_name,
                            "ann_json": ann_json,
                        },
                    )
                    raise e
                try:
                    dataset_fs.add_item_file(
                        video_name,
                        video_file_path,
                        ann=video_ann,
                        _validate_item=False,
                        _use_hardlink=True,
                        item_info=item_info,
                    )
                except Exception as e:
                    logger.info(
                        "INFO FOR DEBUGGING",
                        extra={
                            "project_id": project_id,
                            "dataset_id": dataset.id,
                            "video_id": video_id,
                            "video_name": video_name,
                            "video_file_path": video_file_path,
                            "item_info": item_info,
                        },
                    )
                    raise e

                if progress_cb is not None:
                    progress_cb(1)

            if log_progress:
                ds_progress(len(batch))

    project_fs.set_key_id_map(key_id_map)


def upload_video_project(
    dir: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: bool = True,
    include_custom_data: Optional[bool] = False,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> Tuple[int, str]:
    project_fs = VideoProject.read_single(dir)
    if project_name is None:
        project_name = project_fs.name

    is_multiview = False
    try:
        if project_fs.meta.labeling_interface == LabelingInterface.MULTIVIEW:
            is_multiview = True
    except AttributeError:
        is_multiview = False

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(workspace_id, project_name, ProjectType.VIDEOS)
    project_meta = api.project.update_meta(project.id, project_fs.meta.to_json())

    if progress_cb is not None:
        log_progress = False

    dataset_map = {}
    for dataset_fs in project_fs.datasets:
        dataset_fs: VideoDataset
        if len(dataset_fs.parents) > 0:
            parent = f"{os.path.sep}".join(dataset_fs.parents)
            parent_id = dataset_map.get(parent)
        else:
            parent = ""
            parent_id = None
        dataset = api.dataset.create(project.id, dataset_fs.short_name, parent_id=parent_id)
        dataset_map[os.path.join(parent, dataset.name)] = dataset.id

        names, item_paths, ann_paths, metas = [], [], [], []
        for item_name in dataset_fs:
            video_path, ann_path = dataset_fs.get_item_paths(item_name)
            names.append(item_name)
            item_paths.append(video_path)
            ann_paths.append(ann_path)

            # Read video metadata from metadata folder (includes offset for multiview)
            meta = None
            metadata_path = os.path.join(dataset_fs.metadata_directory, f"{item_name}.meta.json")
            if os.path.exists(metadata_path):
                try:
                    meta = load_json_file(metadata_path)
                except Exception:
                    pass
            metas.append(meta)

        if len(item_paths) == 0:
            continue

        ds_progress = progress_cb
        if log_progress is True:
            ds_progress = tqdm_sly(
                desc="Uploading videos to {!r}".format(dataset.name),
                total=len(item_paths),
                position=0,
            )
        try:
            item_infos = api.video.upload_paths(
                dataset.id, names, item_paths, ds_progress, metas=metas
            )
            video_ids = [item_info.id for item_info in item_infos]
            if include_custom_data:
                for item_info in item_infos:
                    item_name = item_info.name
                    custom_data_path = os.path.join(
                        dir, dataset_fs.name, "custom_data", f"{item_name}.json"
                    )

                    if os.path.exists(custom_data_path):
                        custom_data = load_json_file(custom_data_path)
                        api.video.update_custom_data(item_info.id, custom_data)

        except Exception as e:
            logger.info(
                "INFO FOR DEBUGGING",
                extra={
                    "project_id": project.id,
                    "dataset_id": dataset.id,
                    "names": names,
                    "item_paths": item_paths,
                },
            )
            raise e

        anns_progress = None
        if log_progress or progress_cb is not None:
            anns_progress = tqdm_sly(
                desc="Uploading annotations to {!r}".format(dataset.name),
                total=len(video_ids),
                leave=False,
            )
        try:
            if is_multiview:
                api.video.annotation.upload_paths_multiview(
                    video_ids, ann_paths, project_meta, anns_progress
                )
            else:
                api.video.annotation.upload_paths(
                    video_ids, ann_paths, project_fs.meta, anns_progress
                )
        except Exception as e:
            logger.info(
                "INFO FOR DEBUGGING",
                extra={
                    "project_id": project.id,
                    "dataset_id": dataset.id,
                    "item_ids": video_ids,
                    "ann_paths": ann_paths,
                },
            )
            raise e

    return project.id, project.name


async def download_video_project_async(
    api: Api,
    project_id: int,
    dest_dir: str,
    semaphore: Optional[Union[asyncio.Semaphore, int]] = None,
    dataset_ids: Optional[List[int]] = None,
    download_videos: Optional[bool] = True,
    save_video_info: Optional[bool] = False,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    include_custom_data: Optional[bool] = False,
    resume_download: Optional[bool] = False,
    **kwargs,
) -> None:
    """
    Download video project to the local directory.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Project ID to download
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param semaphore: Semaphore to limit the number of simultaneous downloads of items.
    :type semaphore: asyncio.Semaphore or int, optional
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded. Datasets could be downloaded from different projects but with the same data type.
    :type dataset_ids: list(int), optional
    :param download_videos: Include videos in the download.
    :type download_videos: bool, optional
    :param save_video_info: Include video info in the download.
    :type save_video_info: bool, optional
    :param log_progress: Show downloading logs in the output.
    :type log_progress: bool
    :param progress_cb: Function for tracking the download progress.
    :type progress_cb: tqdm or callable, optional
    :param include_custom_data: Include custom data in the download.
    :type include_custom_data: bool, optional
    :return: None.
    :rtype: NoneType
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        from tqdm import tqdm
        import supervisely as sly

        os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
        os.environ['API_TOKEN'] = 'Your Supervisely API Token'
        api = sly.Api.from_env()

        dest_dir = 'your/local/dest/dir'
        project_id = 17758

        loop = sly.utils.get_or_create_event_loop()
        loop.run_until_complete(
                        sly.download_async(api, project_id, dest_dir)
                    )
    """
    if semaphore is None:
        semaphore = api.get_default_semaphore()
    elif isinstance(semaphore, int):
        semaphore = asyncio.Semaphore(semaphore)

    key_id_map = KeyIdMap()

    project_fs = None
    meta = ProjectMeta.from_json(api.project.get_meta(project_id, with_settings=True))
    if os.path.exists(dest_dir) and resume_download:
        dump_json_file(meta.to_json(), os.path.join(dest_dir, "meta.json"))
        try:
            project_fs = VideoProject(dest_dir, OpenMode.READ)
        except RuntimeError as e:
            if "Project is empty" in str(e):
                clean_dir(dest_dir)
                project_fs = None
            else:
                raise
    if project_fs is None:
        project_fs = VideoProject(dest_dir, OpenMode.CREATE)
    project_fs.set_meta(meta)

    if progress_cb is not None:
        log_progress = False

    dataset_ids = set(dataset_ids) if (dataset_ids is not None) else None
    for parents, dataset in api.dataset.tree(project_id):
        if dataset_ids is not None and dataset.id not in dataset_ids:
            continue

        dataset_path = Dataset._get_dataset_path(dataset.name, parents)

        existing_datasets = {dataset.path: dataset for dataset in project_fs.datasets}
        if dataset_path in existing_datasets:
            dataset_fs = existing_datasets[dataset_path]
        else:
            dataset_fs = project_fs.create_dataset(dataset.name, dataset_path)
        videos = api.video.get_list(dataset.id)

        if log_progress is True:
            progress_cb = tqdm_sly(
                desc="Downloading videos from {!r}".format(dataset.name),
                total=len(videos),
            )

        tasks = []
        for video in videos:
            task = _download_project_item_async(
                api=api,
                video=video,
                semaphore=semaphore,
                dataset=dataset,
                dest_dir=dest_dir,
                project_fs=project_fs,
                key_id_map=key_id_map,
                dataset_fs=dataset_fs,
                include_custom_data=include_custom_data,
                download_videos=download_videos,
                save_video_info=save_video_info,
                progress_cb=progress_cb,
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    project_fs.set_key_id_map(key_id_map)


def _log_warning(
    video: VideoInfo,
    video_file_path: Optional[str] = None,
    ann_json: Optional[dict] = None,
    item_info: Optional[dict] = None,
):
    """
    This function logs a warning with additional information for debugging.
    It is used in the _download_project_item_async function.
    """
    logger.info(
        "INFO FOR DEBUGGING",
        extra={
            "project_id": video.project_id,
            "dataset_id": video.dataset_id,
            "video_id": video.id,
            "video_name": video.name,
            "video_file_path": video_file_path,
            "ann_json": ann_json,
            "item_info": item_info,
        },
    )


async def _download_project_item_async(
    api: Api,
    video: VideoInfo,
    semaphore: Union[asyncio.Semaphore, int],
    dataset: DatasetInfo,
    dest_dir: str,
    project_fs: Project,
    key_id_map: KeyIdMap,
    dataset_fs: VideoDataset,
    include_custom_data: bool,
    download_videos: bool,
    save_video_info: bool = False,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> None:
    """
    This function downloads a video item from the project in Supervisely platform asynchronously.
    """

    if isinstance(semaphore, int):
        semaphore = asyncio.Semaphore(semaphore)

    try:
        ann_json = await api.video.annotation.download_async(video.id, video, semaphore=semaphore)
        ann_json = ann_json[0]
    except Exception as e:
        _log_warning(video)
        raise e

    video_file_path = dataset_fs.generate_item_path(video.name)

    if include_custom_data:
        CUSTOM_DATA_DIR = os.path.join(dest_dir, dataset.name, "custom_data")
        mkdir(CUSTOM_DATA_DIR)
        custom_data_path = os.path.join(CUSTOM_DATA_DIR, f"{video.name}.json")
        await dump_json_file_async(video.custom_data, custom_data_path)

    if download_videos:
        try:
            video_file_size = video.file_meta.get("size")
            if progress_cb is not None and video_file_size is not None:
                item_progress = tqdm_sly(
                    desc=f"Downloading '{video.name}'",
                    total=int(video_file_size),
                    unit="B",
                    unit_scale=True,
                    leave=False,
                )
                await api.video.download_path_async(
                    video.id,
                    video_file_path,
                    semaphore=semaphore,
                    progress_cb=item_progress,
                    progress_cb_type="size",
                )
            else:
                await api.video.download_path_async(video.id, video_file_path, semaphore=semaphore)
        except Exception as e:
            _log_warning(video, video_file_path)
            raise e
    else:
        await touch_async(video_file_path)
    item_info = video._asdict() if save_video_info else None

    try:
        video_ann = VideoAnnotation.from_json(ann_json, project_fs.meta, key_id_map)
    except Exception as e:
        _log_warning(video, ann_json=ann_json)
        raise e
    try:
        await dataset_fs.add_item_file_async(
            video.name,
            None,
            ann=video_ann,
            _validate_item=False,
            _use_hardlink=True,
            item_info=item_info,
        )
    except Exception as e:
        _log_warning(video, video_file_path, item_info)
        raise e

    if progress_cb is not None:
        progress_cb(1)
