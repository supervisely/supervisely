# coding: utf-8

# docs
from collections import namedtuple
import os
from typing import List, Optional, Tuple
from supervisely.api.api import Api


from supervisely.io.fs import file_exists, touch
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress
from supervisely._utils import batched
from supervisely.video_annotation.key_id_map import KeyIdMap

from supervisely.api.module_api import ApiField
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.video import video as sly_video

from supervisely.project.project import Dataset, Project, OpenMode
from supervisely.project.project import read_single_project as read_project_wrapper
from supervisely.project.project_type import ProjectType
from supervisely.video_annotation.video_annotation import VideoAnnotation

VideoItemPaths = namedtuple("VideoItemPaths", ["video_path", "ann_path"])


class VideoDataset(Dataset):
    """
    This is a class for creating and using VideoDataset objects. Here is where your labeled and unlabeled images and other
    files live. There is no more levels: images or videos are directly attached to a dataset. Dataset is a unit of work.
    All images or videos are directly attached to a dataset. A dataset is some sort of data folder with stuff to annotate.
    """

    item_dir_name = "video"
    annotation_class = VideoAnnotation
    item_module = sly_video
    paths_tuple = VideoItemPaths

    @classmethod
    def _has_valid_ext(cls, path: str) -> bool:
        """
        Checks if file from given path is supported
        :param path: str
        :return: bool
        """
        return cls.item_module.has_valid_ext(path)

    def _get_empty_annotaion(self, item_name):
        """
        Create empty VideoAnnotation for given video
        :param item_name: str
        :return: VideoAnnotation class object
        """
        (
            img_size,
            frames_count,
        ) = VideoDataset.item_module.get_image_size_and_frames_count(item_name)
        return self.annotation_class(img_size, frames_count)

    def add_item_np(self, item_name, img, ann=None):
        raise RuntimeError("Deprecated method. Works only with images project")

    def _add_img_np(self, item_name, img):
        raise RuntimeError("Deprecated method. Works only with images project")

    @staticmethod
    def _validate_added_item_or_die(item_path):
        """
        Make sure we actually received a valid image file, clean it up and fail if not so.
        :param item_path: str
        """
        # Make sure we actually received a valid image file, clean it up and fail if not so.
        try:
            VideoDataset.item_module.validate_format(item_path)
        except Exception as e:
            os.remove(item_path)
            raise e

    def get_item_paths(self, item_name) -> VideoItemPaths:
        """
        :param item_name: str
        :return: namedtuple object containing paths to video and annotation from given path
        """
        return VideoDataset.paths_tuple(
            self.get_img_path(item_name), self.get_ann_path(item_name)
        )


class VideoProject(Project):
    """
    This is a class for creating and using VideoProject objects. You can think of a VideoProject as a superfolder with data and
    meta information.
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

    def _read(self):
        """
        Download project from given project directory. Checks item and annotation directoris existing and dataset not empty.
        Consistency checks. Every video must have an annotation, and the correspondence must be one to one.
        """
        super()._read()
        self._key_id_map = KeyIdMap()
        if os.path.exists(self._get_key_id_map_path()):
            self._key_id_map.load_json(self._get_key_id_map_path())

    def _create(self):
        """
        Creates a leaf directory and empty meta.json file. Generate exception error if project directory already exists and is not empty.
        """
        super()._create()
        self.set_key_id_map(KeyIdMap())

    def _add_item_file_to_dataset(
        self, ds, item_name, item_paths, _validate_item, _use_hardlink
    ):
        """
        Add given item file to dataset items directory and add annatation to dataset annotations dir corresponding to item.
        Generate exception error if item_name already exists in dataset or item name has unsupported extension
        :param ds: VideoDataset class object
        :param item_name: str
        :param item_paths: ItemPaths object
        :param _validate_item: bool
        :param _use_hardlink: bool
        """
        ds.add_item_file(
            item_name,
            item_paths.item_path,
            ann=item_paths.ann_path,
            _validate_item=_validate_item,
            _use_hardlink=_use_hardlink,
        )

    @property
    def key_id_map(self):
        return self._key_id_map

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

    @classmethod
    def read_single(cls, dir):
        """
        Read project from given ditectory. Generate exception error if given dir contains more than one subdirectory
        :param dir: str
        :return: VideoProject class object
        """
        return read_project_wrapper(dir, cls)


def download_video_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: List[int] = None,
    download_videos: bool = True,
    log_progress: bool = False,
) -> None:
    """
    Download project with given id in destination directory.

    :param api: Api class object.
    :type api: Api
    :param project_id: Project ID in Supervisely.
    :type project_id: int
    :param dest_dir: Directory to download video project.
    :type dest_dir: str
    :param dataset_ids: Datasets IDs in Supervisely to download.
    :type dataset_ids: List[int], optional
    :param download_videos: Download videos from Supervisely video project in dest_dir or not.
    :type download_videos: bool, optional
    :param log_progress: Logging progress of download video project or not.
    :type log_progress: bool, optional
    :return: None
    :rtype: :class:`NoneType`
    """
    LOG_BATCH_SIZE = 1

    key_id_map = KeyIdMap()

    project_fs = VideoProject(dest_dir, OpenMode.CREATE)

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
        videos = api.video.get_list(dataset.id)

        ds_progress = None
        if log_progress:
            ds_progress = Progress(
                "Downloading dataset: {!r}".format(dataset.name), total_cnt=len(videos)
            )
        for batch in batched(videos, batch_size=LOG_BATCH_SIZE):
            video_ids = [video_info.id for video_info in batch]
            video_names = [video_info.name for video_info in batch]

            ann_jsons = api.video.annotation.download_bulk(dataset.id, video_ids)

            for video_id, video_name, ann_json in zip(
                video_ids, video_names, ann_jsons
            ):
                if video_name != ann_json[ApiField.VIDEO_NAME]:
                    raise RuntimeError(
                        "Error in api.video.annotation.download_batch: broken order"
                    )

                video_file_path = dataset_fs.generate_item_path(video_name)
                if download_videos is True:
                    api.video.download_path(video_id, video_file_path)
                else:
                    touch(video_file_path)

                dataset_fs.add_item_file(
                    video_name,
                    video_file_path,
                    ann=VideoAnnotation.from_json(
                        ann_json, project_fs.meta, key_id_map
                    ),
                    _validate_item=False,
                )

            if log_progress:
                ds_progress.iters_done_report(len(batch))

    project_fs.set_key_id_map(key_id_map)


def upload_video_project(
    dir: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: Optional[bool] = True,
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
    :type log_progress: bool, optional
    :return: New video project ID in Supervisely and project name
    :rtype: :class:`Tuple[int, str]`
    """
    project_fs = VideoProject.read_single(dir)
    if project_name is None:
        project_name = project_fs.name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(workspace_id, project_name, ProjectType.VIDEOS)
    api.project.update_meta(project.id, project_fs.meta.to_json())

    for dataset_fs in project_fs.datasets:
        dataset = api.dataset.create(project.id, dataset_fs.name)

        names, item_paths, ann_paths = [], [], []
        for item_name in dataset_fs:
            img_path, ann_path = dataset_fs.get_item_paths(item_name)
            names.append(item_name)
            item_paths.append(img_path)
            ann_paths.append(ann_path)

        progress_cb = None
        if log_progress:
            ds_progress = Progress(
                "Uploading videos to dataset {!r}".format(dataset.name),
                total_cnt=len(item_paths),
            )
            progress_cb = ds_progress.iters_done_report

        item_infos = api.video.upload_paths(dataset.id, names, item_paths, progress_cb)
        item_ids = [item_info.id for item_info in item_infos]
        if log_progress:
            ds_progress = Progress(
                "Uploading annotations to dataset {!r}".format(dataset.name),
                total_cnt=len(item_paths),
            )
            progress_cb = ds_progress.iters_done_report

        api.video.annotation.upload_paths(
            item_ids, ann_paths, project_fs.meta, progress_cb
        )

    return project.id, project.name
