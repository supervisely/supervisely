# coding: utf-8

# docs
import os
from collections import namedtuple
from supervisely.api.api import Api
from supervisely.annotation.annotation import Annotation
from typing import Tuple, List, Dict, Optional, Callable

from supervisely._utils import batched
from supervisely.api.module_api import ApiField
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.io.fs import touch, dir_exists, list_files, mkdir
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.pointcloud_annotation.pointcloud_episode_annotation import PointcloudEpisodeAnnotation
from supervisely.project.pointcloud_project import PointcloudProject, PointcloudDataset
from supervisely.project.project import OpenMode
from supervisely.project.project import read_single_project as read_project_wrapper
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.project.project_type import ProjectType

PointcloudItemPaths = namedtuple('PointcloudItemPaths', ['pointcloud_path', 'related_images_dir'])


class PointcloudEpisodeDataset(PointcloudDataset):
    item_dir_name = 'pointcloud'
    related_images_dir_name = 'related_images'
    annotation_class = PointcloudEpisodeAnnotation

    def get_item_paths(self, item_name: str) -> PointcloudItemPaths:
        return PointcloudItemPaths(pointcloud_path=self.get_img_path(item_name),
                                   related_images_dir=self.get_related_images_path(item_name))

    def get_ann_path(self) -> str:
        return os.path.join(self.directory, "annotation.json")

    def get_frame_pointcloud_map_path(self) -> str:
        return os.path.join(self.directory, "frame_pointcloud_map.json")

    def set_ann(self, ann: Annotation) -> None:
        if type(ann) is not self.annotation_class:
            raise TypeError("Type of 'ann' have to be Annotation, not a {}".format(type(ann)))
        dst_ann_path = self.get_ann_path()
        dump_json_file(ann.to_json(), dst_ann_path)

    def _add_ann_by_type(self, item_name, ann):
        return

    def _add_img_info(self, item_name, img_info=None):
        return

    def _create(self):
        mkdir(self.item_dir)

    def _read(self):
        if not dir_exists(self.item_dir):
            raise NotADirectoryError(f"Cannot read dataset {self.item_dir}: directory not found")

        try:
            item_paths = sorted(list_files(self.item_dir, filter_fn=self._has_valid_ext))
            item_names = sorted([os.path.basename(path) for path in item_paths])

            map_file_path = self.get_frame_pointcloud_map_path()
            if os.path.isfile(map_file_path):
                self._frame_to_pc_map = load_json_file(map_file_path)
            else:
                self._frame_to_pc_map = {frame_index: item_names[frame_index] for frame_index in range(len(item_names))}

            self._pc_to_frame = {v: k for k, v in self._frame_to_pc_map.items()}
            self._item_to_ann = {name: self._pc_to_frame[name] for name in item_names}
        except Exception as ex:
            raise Exception(f"Cannot read dataset ({self.item_dir}): {repr(ex)}")

    def get_frame_idx(self, item_name: str) -> int:
        return int(self._item_to_ann[item_name])


class PointcloudEpisodeProject(PointcloudProject):
    dataset_class = PointcloudEpisodeDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = PointcloudEpisodeDataset

    @classmethod
    def read_single(cls, dir):
        return read_project_wrapper(dir, cls)


def download_pointcloud_episode_project(api: Api, project_id: int, dest_dir: str, dataset_ids: Optional[List[int]]=None,
                                        download_pcd: Optional[bool]=True,
                                        download_related_images: Optional[bool]=True,
                                        download_annotations: Optional[bool]=True,
                                        log_progress: Optional[bool]=False, batch_size: Optional[int]=10,
                                        progress_cb: Optional[Callable] = None) -> None:
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
            ds_progress = Progress('Downloading dataset: {!r}'.format(dataset.name), total_cnt=len(pointclouds))

        for batch in batched(pointclouds, batch_size=batch_size):
            pointcloud_ids = [pointcloud_info.id for pointcloud_info in batch]
            pointcloud_names = [pointcloud_info.name for pointcloud_info in batch]

            for pointcloud_id, pointcloud_name in zip(pointcloud_ids, pointcloud_names):
                pointcloud_file_path = dataset_fs.generate_item_path(pointcloud_name)
                if download_pcd is True:
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

                dataset_fs.add_item_file(pointcloud_name,
                                         pointcloud_file_path,
                                         _validate_item=False)
            if progress_cb is not None:
                progress_cb(len(batch))
            if log_progress:
                ds_progress.iters_done_report(len(batch))

    project_fs.set_key_id_map(key_id_map)


def upload_pointcloud_episode_project(directory: str, api: Api, workspace_id: int, project_name: Optional[str]=None,
                                      log_progress: Optional[bool]=False) -> Tuple[int, str]:
    # STEP 0 — create project remotely
    project_locally = PointcloudEpisodeProject(directory, OpenMode.READ)
    project_name = project_locally.name if project_name is None else project_name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project_remotely = api.project.create(workspace_id, project_name, ProjectType.POINT_CLOUD_EPISODES)
    api.project.update_meta(project_remotely.id, project_locally.meta.to_json())

    uploaded_objects = KeyIdMap()
    for dataset_locally in project_locally.datasets:
        ann_json_path = dataset_locally.get_ann_path()

        if os.path.isfile(ann_json_path):
            ann_json = load_json_file(ann_json_path)
            episode_annotation = PointcloudEpisodeAnnotation.from_json(ann_json, project_locally.meta)
        else:
            episode_annotation = PointcloudEpisodeAnnotation()

        dataset_remotely = api.dataset.create(project_remotely.id,
                                              dataset_locally.name,
                                              description=episode_annotation.description,
                                              change_name_if_conflict=True)

        # STEP 1 — upload episodes
        items_infos = {
            'names': [],
            'paths': [],
            'metas': []
        }

        for item_name in dataset_locally:
            item_path, related_images_dir = dataset_locally.get_item_paths(item_name)
            frame_idx = dataset_locally.get_frame_idx(item_name)

            item_meta = {
                "frame": frame_idx
            }

            items_infos['names'].append(item_name)
            items_infos['paths'].append(item_path)
            items_infos['metas'].append(item_meta)

        ds_progress = Progress('Uploading pointclouds: {!r}'.format(dataset_remotely.name),
                               total_cnt=len(dataset_locally)) if log_progress else None
        pcl_infos = api.pointcloud_episode.upload_paths(dataset_remotely.id,
                                                        names=items_infos['names'],
                                                        paths=items_infos['paths'],
                                                        metas=items_infos['metas'],
                                                        progress_cb=ds_progress.iters_done_report if log_progress else None)

        # STEP 2 — upload annotations
        frame_to_pcl_ids = {pcl_info.frame: pcl_info.id for pcl_info in pcl_infos}
        api.pointcloud_episode.annotation.append(dataset_remotely.id,
                                                 episode_annotation,
                                                 frame_to_pcl_ids,
                                                 uploaded_objects)

        # STEP 3 — upload photo context
        img_infos = {
            'img_paths': [],
            'img_metas': []
        }

        # STEP 3.1 — upload images
        for pcl_info in pcl_infos:
            related_items = dataset_locally.get_related_images(pcl_info.name)
            images_paths_for_frame = [img_path for img_path, _ in related_items]

            img_infos['img_paths'].extend(images_paths_for_frame)

        img_progress = Progress('Uploading photo context: {!r}'.format(dataset_remotely.name),
                                total_cnt=len(img_infos['img_paths'])) if log_progress else None

        images_hashes = api.pointcloud_episode.upload_related_images(img_infos['img_paths'],
                                                                     progress_cb=img_progress.iters_done_report if log_progress else None)

        # STEP 3.2 — upload images metas
        images_hashes_iterator = images_hashes.__iter__()
        for pcl_info in pcl_infos:
            related_items = dataset_locally.get_related_images(pcl_info.name)

            for _, meta_json in related_items:
                img_hash = next(images_hashes_iterator)
                img_infos['img_metas'].append({ApiField.ENTITY_ID: pcl_info.id,
                                               ApiField.NAME: meta_json[ApiField.NAME],
                                               ApiField.HASH: img_hash,
                                               ApiField.META: meta_json[ApiField.META]})

        if len(img_infos['img_metas']) > 0:
            api.pointcloud_episode.add_related_images(img_infos['img_metas'])

    return project_remotely.id, project_remotely.name
