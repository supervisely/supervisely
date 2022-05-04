# coding: utf-8

from collections import namedtuple
import os

from supervisely.io.fs import file_exists, touch
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress
from supervisely._utils import batched
from supervisely.video_annotation.key_id_map import KeyIdMap

from supervisely.api.api import Api
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
    interpolation_dir = "interpolation"
    annotation_class = VolumeAnnotation
    item_module = sly_volume
    paths_tuple = VolumeItemPaths

    def _get_empty_annotaion(self, item_name):
        path = item_name
        _, volume_meta = sly_volume.read_nrrd_serie_volume(path)
        return self.annotation_class(volume_meta)

    def get_interpolation_dir(self, item_name):
        return os.path.join(self.directory, self.interpolation_dir, item_name)

    def get_interpolation_path(self, item_name, figure):
        return os.path.join(
            self.get_interpolation_dir(item_name), figure.key().hex + ".stl"
        )


class VolumeProject(VideoProject):
    dataset_class = VolumeDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = VolumeDataset


def download_volume_project(
    api: Api,
    project_id,
    dest_dir,
    dataset_ids=None,
    download_volumes=True,
    log_progress=False,
):
    LOG_BATCH_SIZE = 1

    key_id_map = KeyIdMap()

    project_fs = VolumeProject(dest_dir, OpenMode.CREATE)

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
        volumes = api.volume.get_list(dataset.id)

        ds_progress = None
        if log_progress:
            ds_progress = Progress(
                "Downloading dataset: {!r}".format(dataset.name), total_cnt=len(volumes)
            )
        for batch in batched(volumes, batch_size=LOG_BATCH_SIZE):
            volume_ids = [volume_info.id for volume_info in batch]
            volume_names = [volume_info.name for volume_info in batch]

            ann_jsons = api.volume.annotation.download_bulk(dataset.id, volume_ids)

            for volume_id, volume_name, volume_info, ann_json in zip(
                volume_ids, volume_names, batch, ann_jsons
            ):
                if volume_name != ann_json[ApiField.VOLUME_NAME]:
                    raise RuntimeError(
                        "Error in api.volume.annotation.download_batch: broken order"
                    )

                volume_file_path = dataset_fs.generate_item_path(volume_name)
                if download_volumes is True:
                    item_progress = None
                    if log_progress:
                        item_progress = Progress(
                            f"Downloading {volume_name}",
                            total_cnt=volume_info.sizeb,
                            is_size=True,
                        )
                    api.volume.download_path(
                        volume_id, volume_file_path, item_progress.iters_done_report
                    )
                else:
                    touch(volume_file_path)

                ann = VolumeAnnotation.from_json(ann_json, project_fs.meta, key_id_map)
                dataset_fs.add_item_file(
                    volume_name,
                    volume_file_path,
                    ann=ann,
                    _validate_item=False,
                )

                mesh_ids = []
                mesh_paths = []
                for sf in ann.spatial_figures:
                    figure_id = key_id_map.get_figure_id(sf.key())
                    mesh_ids.append(figure_id)
                    figure_path = dataset_fs.get_interpolation_path(volume_name, sf)
                    mesh_paths.append(figure_path)
                api.volume.figure.download_stl_meshes(mesh_ids, mesh_paths)

            ds_progress.iters_done_report(len(batch))
    project_fs.set_key_id_map(key_id_map)


# TODO: add methods to convert to 3d masks


def upload_volume_project(
    dir, api: Api, workspace_id, project_name=None, log_progress=True
):
    project_fs = VolumeProject.read_single(dir)
    if project_name is None:
        project_name = project_fs.name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(workspace_id, project_name, ProjectType.VOLUMES)
    api.project.update_meta(project.id, project_fs.meta.to_json())

    for dataset_fs in project_fs.datasets:
        dataset = api.dataset.create(project.id, dataset_fs.name)

        names, item_paths, ann_paths, interpolation_dirs = [], [], [], []
        for item_name in dataset_fs:
            img_path, ann_path = dataset_fs.get_item_paths(item_name)
            names.append(item_name)
            item_paths.append(img_path)
            ann_paths.append(ann_path)
            interpolation_dirs.append(dataset_fs.get_interpolation_dir(item_name))

        progress_cb = None
        if log_progress:
            ds_progress = Progress(
                "Uploading volumes to dataset {!r}".format(dataset.name),
                total_cnt=len(item_paths),
            )
            progress_cb = ds_progress.iters_done_report

        item_infos = api.volume.upload_nrrd_series_paths(
            dataset.id, names, item_paths, progress_cb
        )
        item_ids = [item_info.id for item_info in item_infos]
        ds_progress = None
        if log_progress:
            ds_progress = Progress(
                "Uploading annotations to dataset {!r}".format(dataset.name),
                total_cnt=len(item_paths),
            )
            progress_cb = ds_progress.iters_done_report

        api.volume.annotation.upload_paths(
            item_ids, ann_paths, project_fs.meta, interpolation_dirs, progress_cb
        )

    return project.id, project.name
