# coding: utf-8

from collections import namedtuple
import os

from supervisely_lib.io.fs import file_exists, touch
from supervisely_lib.io.json import dump_json_file, load_json_file
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.task.progress import Progress
from supervisely_lib._utils import batched
from supervisely_lib.video_annotation.key_id_map import KeyIdMap

from supervisely_lib.api.module_api import ApiField
from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection
from supervisely_lib.volume import volume as sly_volume
from supervisely_lib.project.project import Dataset, Project, OpenMode
from supervisely_lib.project.project import read_single_project as read_project_wrapper
from supervisely_lib.project.project_type import ProjectType
from supervisely_lib.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely_lib.volume.volume import validate_format

VolumeItemPaths = namedtuple('VolumeItemPaths', ['volume_path', 'ann_path'])


class VolumeDataset(Dataset):
    item_dir_name = 'volume'
    annotation_class = VolumeAnnotation

    @staticmethod
    def _has_valid_ext(path: str) -> bool:
        return sly_volume.has_valid_ext(path)

    def _get_empty_annotaion(self, item_name):
        return self.annotation_class()

    def add_item_np(self, item_name, img, ann=None):
        raise RuntimeError("Deprecated method. Works only with images project")

    def _add_img_np(self, item_name, img):
        raise RuntimeError("Deprecated method. Works only with images project")

    @staticmethod
    def _validate_added_item_or_die(item_path):
        try:
            sly_volume.validate_format(item_path)
        except (sly_volume.UnsupportedVolumeFormat, sly_volume.VolumeReadException) as e:
            os.remove(item_path)
            raise e

    def set_ann(self, item_name: str, ann):
        if type(ann) is not self.annotation_class:
            raise TypeError(f"Type of 'ann' have to be {self.annotation_class}, not a {ann}")

        dst_ann_path = self.get_ann_path(item_name)
        dump_json_file(ann.to_json(), dst_ann_path)

    def get_item_paths(self, item_name) -> VolumeItemPaths:
        volume_path = self.get_img_path(item_name)
        validate_format(volume_path)
        return VolumeItemPaths(volume_path=self.get_img_path(item_name), ann_path=self.get_ann_path(item_name))


class VolumeProject(Project):
    dataset_class = VolumeDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = VolumeDataset

    def __init__(self, directory, mode: OpenMode):
        self._key_id_map: KeyIdMap = None
        super().__init__(directory, mode)

    def _read(self):
        super(VolumeProject, self)._read()
        self._key_id_map = KeyIdMap()
        self._key_id_map.load_json(self._get_key_id_map_path())

    def _create(self):
        super()._create()
        self.set_key_id_map(KeyIdMap())

    def _add_item_file_to_dataset(self, ds, item_name, item_paths, _validate_item, _use_hardlink):
        ds.add_item_file(item_name, item_paths.item_path,
                         ann=item_paths.ann_path, _validate_item=_validate_item, _use_hardlink=_use_hardlink)

    @property
    def key_id_map(self):
        return self._key_id_map

    def set_key_id_map(self, new_map: KeyIdMap):
        self._key_id_map = new_map
        self._key_id_map.dump_json(self._get_key_id_map_path())

    def _get_key_id_map_path(self):
        return os.path.join(self.directory, 'key_id_map.json')

    @classmethod
    def read_single(cls, dir):
        return read_project_wrapper(dir, cls)


def download_volume_project(api, project_id, dest_dir, dataset_ids=None, download_volumes=True, batch_size=10, log_progress=False):
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
            ds_progress = Progress('Downloading dataset: {!r}'.format(dataset.name), total_cnt=len(volumes))
        for batch in batched(volumes, batch_size=batch_size):
            volume_ids = [volume_info.id for volume_info in batch]
            volume_names = [volume_info.name for volume_info in batch]
            ann_jsons = api.volume.annotation.download_bulk(dataset.id, volume_ids)

            for volume_id, volume_name, ann_json in zip(volume_ids, volume_names, ann_jsons):
                if volume_name != ann_json[ApiField.VOLUME_NAME]:
                    raise RuntimeError("Error in api.volume.annotation.download_batch: broken order")

                volume_file_path = dataset_fs.generate_item_path(volume_name)
                if download_volumes is True:
                    api.volume.download_path(volume_id, volume_file_path)
                else:
                    touch(volume_file_path)

                dataset_fs.add_item_file(volume_name,
                                         volume_file_path,
                                         ann=VolumeAnnotation.from_json(ann_json, project_fs.meta, key_id_map),
                                         _validate_item=False)
            if log_progress:
                ds_progress.iters_done_report(len(batch))

    project_fs.set_key_id_map(key_id_map)


def upload_volume_project(directory, api, workspace_id, project_name=None, progress_cb=None):
    project_fs = VolumeProject(directory, mode=OpenMode.READ)
    if project_name is None:
       project_name = project_fs.name

    if api.project.exists(workspace_id, project_name):
       project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(workspace_id, project_name, ProjectType.VOLUMES)
    api.project.update_meta(project.id, project_fs.meta.to_json())

    uploaded_objects = KeyIdMap()
    for dataset_fs in project_fs:
        dataset = api.dataset.create(project.id, dataset_fs.name, change_name_if_conflict=True)

        anns, item_names, volume_paths, volume_metas = [], [], [], []
        for item_name in dataset_fs:
            volume_path, ann_path = dataset_fs.get_item_paths(item_name)
            ann = VolumeAnnotation.from_json(load_json_file(ann_path), project_fs.meta)
            anns.append(ann)
            item_names.append(item_name)
            volume_paths.append(volume_path)
            volume_metas.append(ann.volume_meta)

        volumes = api.volume.upload_paths(dataset.id, item_names, volume_paths, volume_metas, progress_cb=progress_cb)
        api.volume.get_list(volumes[0].dataset_id)  # TODO: remove - workaround HACK to _get_by_id info

        for i, ann in enumerate(anns):
            api.volume.annotation.append(volumes[i].id, ann, uploaded_objects)

