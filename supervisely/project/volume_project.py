# coding: utf-8

from collections import namedtuple
import os

from supervisely.io.fs import file_exists, touch
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress
from supervisely._utils import batched
from supervisely.video_annotation.key_id_map import KeyIdMap

from supervisely.api.module_api import ApiField
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.volume import volume as sly_volume

from supervisely.project.project import Dataset, Project, OpenMode
from supervisely.project.video_project import VideoDataset, VideoProject
from supervisely.project.project import read_single_project as read_project_wrapper
from supervisely.project.project_type import ProjectType
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation

VolumeItemPaths = namedtuple("VolumeItemPaths", ["volume_path", "ann_path"])


class VolumeDataset(VideoDataset):
    item_dir_name = "volume"
    annotation_class = VolumeAnnotation
    item_module = sly_volume
    paths_tuple = VolumeItemPaths

    def _get_empty_annotaion(self, item_name):
        path = item_name
        _, volume_meta = sly_volume.read_nrrd_serie_volume(path)
        return self.annotation_class(volume_meta)


class VolumeProject(VideoProject):
    dataset_class = VolumeDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = VolumeDataset


def download_volume_project(
    api,
    project_id,
    dest_dir,
    dataset_ids=None,
    download_volumes=True,
    log_progress=False,
):
    """
    Download project with given id in destination directory
    :param api: Api class object
    :param project_id: int
    :param dest_dir: str
    :param dataset_ids: list of integers
    :param download_videos: bool
    :param log_progress: bool
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

            ds_progress.iters_done_report(len(batch))

    project_fs.set_key_id_map(key_id_map)


def upload_video_project(dir, api, workspace_id, project_name=None, log_progress=True):
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
