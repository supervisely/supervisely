# coding: utf-8

# docs
import os
from collections import namedtuple
from supervisely.api.api import Api
from supervisely.annotation.annotation import Annotation
from typing import Tuple, List, Dict, Optional, Callable, NamedTuple, Union

from supervisely._utils import batched
from supervisely.api.module_api import ApiField
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.io.fs import touch, dir_exists, list_files, mkdir
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.pointcloud_annotation.pointcloud_episode_annotation import (
    PointcloudEpisodeAnnotation,
)
from supervisely.project.pointcloud_project import PointcloudProject, PointcloudDataset
from supervisely.project.project import OpenMode
from supervisely.project.project import read_single_project as read_project_wrapper
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger

PointcloudItemPaths = namedtuple(
    "PointcloudItemPaths", ["pointcloud_path", "related_images_dir", "frame_index"]
)
PointcloudItemInfo = namedtuple(
    "PointcloudItemInfo",
    ["dataset_name", "name", "pointcloud_path", "related_images_dir", "frame_index"],
)


class PointcloudEpisodeDataset(PointcloudDataset):
    item_dir_name = "pointcloud"
    related_images_dir_name = "related_images"
    annotation_class = PointcloudEpisodeAnnotation

    @property
    def ann_dir(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} object don't have correct path for 'ann_dir' property. \
            Use 'get_ann_path()' method instead of this."
        )

    def get_item_paths(self, item_name: str) -> PointcloudItemPaths:
        return PointcloudItemPaths(
            pointcloud_path=self.get_pointcloud_path(item_name),
            related_images_dir=self.get_related_images_path(item_name),
            frame_index=self.get_frame_idx(item_name),
        )

    def get_ann_path(self) -> str:
        return os.path.join(self.directory, "annotation.json")

    def get_frame_pointcloud_map_path(self) -> str:
        return os.path.join(self.directory, "frame_pointcloud_map.json")

    def set_ann(self, ann: PointcloudEpisodeAnnotation) -> None:
        if type(ann) is not self.annotation_class:
            raise TypeError(
                f"Type of 'ann' should be {self.annotation_class.__name__}, not a {type(ann).__name__}"
            )
        dst_ann_path = self.get_ann_path()
        dump_json_file(ann.to_json(), dst_ann_path)

    def _create(self):
        mkdir(self.item_dir)

    def _read(self):
        if not dir_exists(self.item_dir):
            raise NotADirectoryError(
                f"Cannot read dataset {self.name}: {self.item_dir} directory not found"
            )

        try:
            item_paths = sorted(list_files(self.item_dir, filter_fn=self._has_valid_ext))
            item_names = sorted([os.path.basename(path) for path in item_paths])

            map_file_path = self.get_frame_pointcloud_map_path()
            if os.path.isfile(map_file_path):
                self._frame_to_pc_map = load_json_file(map_file_path)
            else:
                self._frame_to_pc_map = {
                    frame_index: item_names[frame_index] for frame_index in range(len(item_names))
                }

            self._pc_to_frame = {v: k for k, v in self._frame_to_pc_map.items()}
            self._item_to_ann = {name: self._pc_to_frame[name] for name in item_names}
        except Exception as ex:
            raise Exception(f"Cannot read dataset ({self.name}): {repr(ex)}")

    def add_item_file(
        self,
        item_name: str,
        item_path: str,
        frame: Optional[Union[str, int]] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
        item_info: Optional[NamedTuple] = None,
    ) -> None:
        """
        Adds given item file to dataset items directory, and adds given annotation to dataset ann directory. if ann is None, creates empty annotation file.

        :param item_name: Item name.
        :type item_name: str
        :param item_path: Path to the item.
        :type item_path: str
        :param frame: Frame number.
        :type frame: str or int, optional
        :param _validate_item: Checks input files format.
        :type _validate_item: bool, optional
        :param _use_hardlink: If True creates a hardlink pointing to src named dst, otherwise don't.
        :type _use_hardlink: bool, optional
        :param item_info: NamedTuple ImageInfo containing information about pointcloud.
        :type item_info: NamedTuple, optional
        :return: None
        :rtype: :class:`NoneType`
        :raises: :class:`Exception` if item_name already exists in dataset or item name has unsupported extension.
        :Usage example:

         .. code-block:: python

            from supervisely.project.pointcloud_episode_project import PointcloudEpisodeDataset
            dataset_path = "/home/admin/work/supervisely/projects/episodes_project/episode_0"
            ds = PointcloudEpisodeDataset(dataset_path, sly.OpenMode.READ)

            ds.add_item_file("PTC_777.pcd", "/home/admin/work/supervisely/projects/episodes_project/episode_0/pointcloud/PTC_777.pcd", frame=3)
        """
        if item_path is None and item_info is None:
            raise RuntimeError("No item_path or ann or item_info provided.")

        self._add_item_file(
            item_name,
            item_path,
            _validate_item=_validate_item,
            _use_hardlink=_use_hardlink,
        )
        self._add_ann_by_type(item_name, frame)
        self._add_pointcloud_info(item_name, item_info)

    def _add_ann_by_type(self, item_name, frame):
        if frame is None:
            self._item_to_ann[item_name] = ""
        elif isinstance(frame, int):
            self._item_to_ann[item_name] = str(frame)
        elif type(frame) is str:
            self._item_to_ann[item_name] = frame
        else:
            raise TypeError("Unsupported type {!r} for ann argument".format(type(frame)))

    def get_frame_idx(self, item_name: str) -> int:
        if self._item_to_ann[item_name] == "":
            return -1
        return int(self._item_to_ann[item_name])

class PointcloudEpisodeProject(PointcloudProject):
    dataset_class = PointcloudEpisodeDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = PointcloudEpisodeDataset

    @classmethod
    def read_single(cls, dir):
        return read_project_wrapper(dir, cls)


def download_pointcloud_episode_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    download_pcd: Optional[bool] = True,
    download_related_images: Optional[bool] = True,
    download_annotations: Optional[bool] = True,
    log_progress: Optional[bool] = False,
    batch_size: Optional[int] = 10,
    save_pointcloud_info: Optional[bool] = False,
    save_pointclouds: Optional[bool] = True,
    progress_cb: Optional[Callable] = None,
) -> PointcloudEpisodeProject:
    key_id_map = KeyIdMap()
    project_fs = PointcloudEpisodeProject(dest_dir, OpenMode.CREATE)
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
        pointclouds = api.pointcloud_episode.get_list(dataset.id)

        if download_annotations:
            # Download annotation to project_path/dataset_path/annotation.json
            ann_json = api.pointcloud_episode.annotation.download(dataset.id)
            annotation = dataset_fs.annotation_class.from_json(ann_json, meta, key_id_map)
            dataset_fs.set_ann(annotation)

            # frames --> pointcloud mapping to project_path/dataset_path/frame_pointcloud_map.json
            frame_name_map = api.pointcloud_episode.get_frame_name_map(dataset.id)
            frame_pointcloud_map_path = dataset_fs.get_frame_pointcloud_map_path()
            dump_json_file(frame_name_map, frame_pointcloud_map_path)

        # Download data
        if log_progress:
            ds_progress = Progress(
                "Downloading dataset: {!r}".format(dataset.name), total_cnt=len(pointclouds)
            )

        for batch in batched(pointclouds, batch_size=batch_size):
            pointcloud_ids = [pointcloud_info.id for pointcloud_info in batch]
            pointcloud_names = [pointcloud_info.name for pointcloud_info in batch]
            map_file_path = dataset_fs.get_frame_pointcloud_map_path()
            frame_to_pc_map = load_json_file(map_file_path)
            pc_to_frame = {v: k for k, v in frame_to_pc_map.items()}
            item_to_ann = {name: pc_to_frame[name] for name in pointcloud_names}

            for pointcloud_id, pointcloud_name, pointcloud_info in zip(
                pointcloud_ids, pointcloud_names, batch
            ):
                pointcloud_file_path = dataset_fs.generate_item_path(pointcloud_name)
                if download_pcd:
                    api.pointcloud_episode.download_path(pointcloud_id, pointcloud_file_path)
                else:
                    touch(pointcloud_file_path)

                if download_related_images:
                    related_images_path = dataset_fs.get_related_images_path(pointcloud_name)
                    related_images = api.pointcloud_episode.get_list_related_images(pointcloud_id)
                    for rimage_info in related_images:
                        name = rimage_info[ApiField.NAME]
                        rimage_id = rimage_info[ApiField.ID]

                        path_img = os.path.join(related_images_path, name)
                        path_json = os.path.join(related_images_path, name + ".json")

                        api.pointcloud_episode.download_related_image(rimage_id, path_img)
                        dump_json_file(rimage_info, path_json)

                dataset_fs.add_item_file(
                    pointcloud_name,
                    pointcloud_file_path if save_pointclouds else None,
                    item_to_ann[pointcloud_name],
                    _validate_item=False,
                    item_info=pointcloud_info if save_pointcloud_info else None,
                )
            if progress_cb is not None:
                progress_cb(len(batch))
            if log_progress:
                ds_progress.iters_done_report(len(batch))

    project_fs.set_key_id_map(key_id_map)
    return project_fs


def upload_pointcloud_episode_project(
    directory: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: Optional[bool] = False,
) -> Tuple[int, str]:
    # STEP 0 — create project remotely
    project_locally = PointcloudEpisodeProject(directory, OpenMode.READ)
    project_name = project_locally.name if project_name is None else project_name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project_remotely = api.project.create(
        workspace_id, project_name, ProjectType.POINT_CLOUD_EPISODES
    )
    api.project.update_meta(project_remotely.id, project_locally.meta.to_json())

    uploaded_objects = KeyIdMap()
    for dataset_locally in project_locally.datasets:
        ann_json_path = dataset_locally.get_ann_path()

        if os.path.isfile(ann_json_path):
            ann_json = load_json_file(ann_json_path)
            episode_annotation = PointcloudEpisodeAnnotation.from_json(
                ann_json, project_locally.meta
            )
        else:
            episode_annotation = PointcloudEpisodeAnnotation()

        dataset_remotely = api.dataset.create(
            project_remotely.id,
            dataset_locally.name,
            description=episode_annotation.description,
            change_name_if_conflict=True,
        )

        # STEP 1 — upload episodes
        items_infos = {"names": [], "paths": [], "metas": []}

        for item_name in dataset_locally:
            item_path, related_images_dir = dataset_locally.get_item_paths(item_name)
            frame_idx = dataset_locally.get_frame_idx(item_name)

            item_meta = {"frame": frame_idx}

            items_infos["names"].append(item_name)
            items_infos["paths"].append(item_path)
            items_infos["metas"].append(item_meta)

        ds_progress = (
            Progress(
                "Uploading pointclouds: {!r}".format(dataset_remotely.name),
                total_cnt=len(dataset_locally),
            )
            if log_progress
            else None
        )
        pcl_infos = api.pointcloud_episode.upload_paths(
            dataset_remotely.id,
            names=items_infos["names"],
            paths=items_infos["paths"],
            metas=items_infos["metas"],
            progress_cb=ds_progress.iters_done_report if log_progress else None,
        )

        # STEP 2 — upload annotations
        frame_to_pcl_ids = {pcl_info.frame: pcl_info.id for pcl_info in pcl_infos}
        api.pointcloud_episode.annotation.append(
            dataset_remotely.id, episode_annotation, frame_to_pcl_ids, uploaded_objects
        )

        # STEP 3 — upload photo context
        img_infos = {"img_paths": [], "img_metas": []}

        # STEP 3.1 — upload images
        for pcl_info in pcl_infos:
            related_items = dataset_locally.get_related_images(pcl_info.name)
            images_paths_for_frame = [img_path for img_path, _ in related_items]

            img_infos["img_paths"].extend(images_paths_for_frame)

        img_progress = (
            Progress(
                "Uploading photo context: {!r}".format(dataset_remotely.name),
                total_cnt=len(img_infos["img_paths"]),
            )
            if log_progress
            else None
        )

        images_hashes = api.pointcloud_episode.upload_related_images(
            img_infos["img_paths"],
            progress_cb=img_progress.iters_done_report if log_progress else None,
        )

        # STEP 3.2 — upload images metas
        images_hashes_iterator = images_hashes.__iter__()
        for pcl_info in pcl_infos:
            related_items = dataset_locally.get_related_images(pcl_info.name)

            for _, meta_json in related_items:
                img_hash = next(images_hashes_iterator)
                img_infos["img_metas"].append(
                    {
                        ApiField.ENTITY_ID: pcl_info.id,
                        ApiField.NAME: meta_json[ApiField.NAME],
                        ApiField.HASH: img_hash,
                        ApiField.META: meta_json[ApiField.META],
                    }
                )

        if len(img_infos["img_metas"]) > 0:
            api.pointcloud_episode.add_related_images(img_infos["img_metas"])

    return project_remotely.id, project_remotely.name
