# coding: utf-8

import os
from collections import namedtuple

from supervisely_lib._utils import batched
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection
from supervisely_lib.io.fs import touch, dir_exists, list_files, mkdir
from supervisely_lib.io.json import dump_json_file, load_json_file
from supervisely_lib.pointcloud_annotation.pointcloud_episode_annotation import PointcloudEpisodeAnnotation
from supervisely_lib.project.pointcloud_project import PointcloudProject, PointcloudDataset
from supervisely_lib.project.project import OpenMode
from supervisely_lib.project.project import read_single_project as read_project_wrapper
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.task.progress import Progress
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.project.project_type import ProjectType

PointcloudItemPaths = namedtuple('PointcloudItemPaths', ['pointcloud_path', 'related_images_dir'])


class PointcloudEpisodeDataset(PointcloudDataset):
    item_dir_name = 'pointcloud'
    related_images_dir_name = 'related_images'
    annotation_class = PointcloudEpisodeAnnotation

    def get_item_paths(self, item_name) -> PointcloudItemPaths:
        return PointcloudItemPaths(pointcloud_path=self.get_img_path(item_name),
                                   related_images_dir=self.get_related_images_path(item_name))

    def get_ann_path(self):
        return os.path.join(self.directory, "annotation.json")

    def get_frame_pointcloud_map_path(self):
        return os.path.join(self.directory, "frame_pointcloud_map.json")

    def set_ann(self, ann):
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
            raise FileNotFoundError('Item directory not found: {!r}'.format(self.item_dir))

        item_paths = list_files(self.item_dir, filter_fn=self._has_valid_ext)
        item_names = [os.path.basename(path) for path in item_paths]
        self._frame_to_pc_map = load_json_file(self.get_frame_pointcloud_map_path())
        self._pc_to_frame = {v: k for k, v in self._frame_to_pc_map.items()}
        self._item_to_ann = {name: self._pc_to_frame[name] for name in item_names}

    def get_frame_idx(self, item_name):
        return int(self._item_to_ann[item_name])


class PointcloudEpisodeProject(PointcloudProject):
    dataset_class = PointcloudEpisodeDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = PointcloudEpisodeDataset

    @classmethod
    def read_single(cls, dir):
        return read_project_wrapper(dir, cls)


def download_pointcloud_episode_project(api, project_id, dest_dir, dataset_ids=None,
                                        download_pcd=True,
                                        download_realated_images=True,
                                        download_annotations=True,
                                        log_progress=False, batch_size=10):
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

                if download_realated_images:
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
            if log_progress:
                ds_progress.iters_done_report(len(batch))

    project_fs.set_key_id_map(key_id_map)


def upload_pointcloud_episode_project(directory, api, workspace_id, project_name=None, log_progress=False):
    project_fs = PointcloudEpisodeProject.read_single(directory)
    if project_name is None:
        project_name = project_fs.name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(workspace_id, project_name, ProjectType.POINT_CLOUD_EPISODES)
    api.project.update_meta(project.id, project_fs.meta.to_json())

    uploaded_objects = KeyIdMap()
    for dataset_fs in project_fs.datasets:
        ann_json = load_json_file(dataset_fs.get_ann_path())
        episode_annotation = PointcloudEpisodeAnnotation.from_json(ann_json, project_fs.meta)

        dataset = api.dataset.create(project.id,
                                     dataset_fs.name,
                                     description=episode_annotation.description,
                                     change_name_if_conflict=True)

        frame_to_pointcloud_ids = {}
        ds_progress = None
        if log_progress:
            ds_progress = Progress('Uploading dataset: {!r}'.format(dataset.name), total_cnt=len(dataset_fs))

        for item_name in dataset_fs:
            item_path, related_images_dir = dataset_fs.get_item_paths(item_name)
            related_items = dataset_fs.get_related_images(item_name)

            item_meta = {}
            try:
                _, meta = related_items[0]
                timestamp = meta[ApiField.META]['timestamp']
                item_meta["timestamp"] = timestamp
            except (KeyError, IndexError):
                pass

            frame_idx = dataset_fs.get_frame_idx(item_name)
            item_meta["frame"] = frame_idx
            pointcloud = api.pointcloud_episode.upload_path(dataset.id, item_name, item_path, item_meta)

            if len(related_items) != 0:
                img_infos = []
                for img_path, meta_json in related_items:
                    img = api.pointcloud_episode.upload_related_image(img_path)[0]
                    img_infos.append({ApiField.ENTITY_ID: pointcloud.id,
                                      ApiField.NAME: meta_json[ApiField.NAME],
                                      ApiField.HASH: img,
                                      ApiField.META: meta_json[ApiField.META]})

                api.pointcloud_episode.add_related_images(img_infos)

            frame_to_pointcloud_ids[frame_idx] = pointcloud.id

            if log_progress:
                ds_progress.iters_done_report(1)

        api.pointcloud_episode.annotation.append(dataset.id,
                                                 episode_annotation,
                                                 frame_to_pointcloud_ids,
                                                 uploaded_objects)

    return project.id, project.name
