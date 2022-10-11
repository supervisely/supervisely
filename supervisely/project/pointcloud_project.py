# coding: utf-8

# dict
from typing import List, Optional, Dict, Tuple, Callable
from supervisely.api.api import Api
from collections import namedtuple
import os

from supervisely.io.fs import file_exists, touch, dir_exists, list_files, get_file_name_with_ext, get_file_name
from supervisely.imaging.image import SUPPORTED_IMG_EXTS, has_valid_ext
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


from supervisely.pointcloud_annotation.pointcloud_annotation import PointcloudAnnotation
from supervisely.pointcloud import pointcloud as sly_pointcloud
from supervisely.project.video_project import VideoDataset, VideoProject
from supervisely.io.json import dump_json_file
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger

PointcloudItemPaths = namedtuple('PointcloudItemPaths', ['pointcloud_path', 'related_images_dir', 'ann_path'])


class PointcloudDataset(VideoDataset):
    item_dir_name = 'pointcloud'
    related_images_dir_name = 'related_images'
    annotation_class = PointcloudAnnotation

    @staticmethod
    def _has_valid_ext(path: str) -> bool:
        return sly_pointcloud.has_valid_ext(path)

    def _get_empty_annotaion(self, item_name):
        return self.annotation_class()

    @staticmethod
    def _validate_added_item_or_die(item_path):
        # Make sure we actually received a valid image file, clean it up and fail if not so.
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
        return PointcloudItemPaths(pointcloud_path=self.get_img_path(item_name),
                                   related_images_dir=self.get_related_images_path(item_name),
                                   ann_path=self.get_ann_path(item_name))

    def get_related_images(self, item_name: str) -> List[Tuple[str, Dict]]:
        results = []
        path = self.get_related_images_path(item_name)
        if dir_exists(path):
            files = list_files(path, SUPPORTED_IMG_EXTS)
            for file in files:
                img_meta_path = os.path.join(path, get_file_name_with_ext(file)+".json")
                img_meta = {}
                if file_exists(img_meta_path):
                    img_meta = load_json_file(img_meta_path)
                    if img_meta[ApiField.NAME] != get_file_name_with_ext(file):
                        raise RuntimeError('Wrong format: name field contains wrong image path')
                results.append((file, img_meta))
        return results


class PointcloudProject(VideoProject):
    dataset_class = PointcloudDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = PointcloudDataset

    @classmethod
    def read_single(cls, dir):
        return read_project_wrapper(dir, cls)


def download_pointcloud_project(api: Api, project_id: int, dest_dir: str, dataset_ids: Optional[List[int]]=None,
                                download_items: Optional[bool]=True, log_progress: Optional[bool]=False,
                                progress_cb: Optional[Callable] = None) -> None:
    LOG_BATCH_SIZE = 1

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
            ds_progress = Progress('Downloading dataset: {!r}'.format(dataset.name), total_cnt=len(pointclouds))
        for batch in batched(pointclouds, batch_size=LOG_BATCH_SIZE):
            pointcloud_ids = [pointcloud_info.id for pointcloud_info in batch]
            pointcloud_names = [pointcloud_info.name for pointcloud_info in batch]

            ann_jsons = api.pointcloud.annotation.download_bulk(dataset.id, pointcloud_ids)

            for pointcloud_id, pointcloud_name, ann_json in zip(pointcloud_ids, pointcloud_names, ann_jsons):
                if pointcloud_name != ann_json[ApiField.NAME]:
                    raise RuntimeError("Error in api.video.annotation.download_batch: broken order")

                pointcloud_file_path = dataset_fs.generate_item_path(pointcloud_name)
                if download_items is True:
                    api.pointcloud.download_path(pointcloud_id, pointcloud_file_path)

                    related_images_path = dataset_fs.get_related_images_path(pointcloud_name)
                    related_images = api.pointcloud.get_list_related_images(pointcloud_id)
                    for rimage_info in related_images:
                        name = rimage_info[ApiField.NAME]

                        if not has_valid_ext(name):
                            new_name = get_file_name(name)  # to fix cases like .png.json
                            if has_valid_ext(new_name):
                                name = new_name
                                rimage_info[ApiField.NAME] = name
                            else:
                                raise RuntimeError('Something wrong with photo context filenames.\
                                                    Please, contact support')

                        rimage_id = rimage_info[ApiField.ID]

                        path_img = os.path.join(related_images_path, name)
                        path_json = os.path.join(related_images_path, name + ".json")

                        api.pointcloud.download_related_image(rimage_id, path_img)
                        dump_json_file(rimage_info, path_json)

                else:
                    touch(pointcloud_file_path)

                dataset_fs.add_item_file(pointcloud_name,
                                         pointcloud_file_path,
                                         ann=PointcloudAnnotation.from_json(ann_json, project_fs.meta, key_id_map),
                                         _validate_item=False)
            if progress_cb is not None:
                progress_cb(len(batch))
            if log_progress:
                ds_progress.iters_done_report(len(batch))

    project_fs.set_key_id_map(key_id_map)


def upload_pointcloud_project(directory: str, api: Api, workspace_id: int, project_name: Optional[str]=None, log_progress: Optional[bool]=False) -> Tuple[int, str]:
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
            ds_progress = Progress('Uploading dataset: {!r}'.format(dataset.name), total_cnt=len(dataset_fs))

        for item_name in dataset_fs:

            item_path, related_images_dir, ann_path = dataset_fs.get_item_paths(item_name)
            related_items = dataset_fs.get_related_images(item_name)

            try:
                _, meta = related_items[0]
                timestamp = meta[ApiField.META]['timestamp']
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
                        rimg_infos.append({ApiField.ENTITY_ID: pointcloud.id,
                                        ApiField.NAME: meta_json[ApiField.NAME],
                                        ApiField.HASH: img,
                                        ApiField.META: meta_json[ApiField.META]})
                    except Exception as e:
                        logger.error("Related images uploading error.", extra={
                            "pointcloud_id": pointcloud.id, 
                            "meta_json": meta_json, 
                            "rel_image_hash": img
                        })
                        raise e
                try:
                    api.pointcloud.add_related_images(rimg_infos)
                except Exception as e:
                    logger.error("Related images adding error.", extra={
                        "pointcloud_id": pointcloud.id, 
                        "rel_images_info": rimg_infos
                    })
                    raise e
            if log_progress:
                ds_progress.iters_done_report(1)
                
    return project.id, project_name
