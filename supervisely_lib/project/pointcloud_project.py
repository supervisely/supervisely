# coding: utf-8

from collections import namedtuple
import os

from supervisely_lib.io.fs import file_exists, touch, dir_exists, list_files, get_file_name_with_ext
from supervisely_lib.imaging.image import SUPPORTED_IMG_EXTS
from supervisely_lib.io.json import dump_json_file, load_json_file
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.task.progress import Progress
from supervisely_lib._utils import batched
from supervisely_lib.video_annotation.key_id_map import KeyIdMap

from supervisely_lib.api.module_api import ApiField
from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection
from supervisely_lib.video import video as sly_video

from supervisely_lib.project.project import Dataset, Project, OpenMode
from supervisely_lib.project.project import read_single_project as read_project_wrapper


from supervisely_lib.pointcloud_annotation.pointcloud_annotation import PointcloudAnnotation
from supervisely_lib.pointcloud import pointcloud as sly_pointcloud
from supervisely_lib.project.video_project import VideoDataset, VideoProject
from supervisely_lib.io.json import dump_json_file


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

    def get_related_images_path(self, item_name):
        item_name_temp = item_name.replace(".", "_")
        rimg_dir = os.path.join(self.directory, self.related_images_dir_name, item_name_temp)
        return rimg_dir

    def get_item_paths(self, item_name) -> PointcloudItemPaths:
        return PointcloudItemPaths(pointcloud_path=self.get_img_path(item_name),
                                   related_images_dir=self.get_related_images_path(item_name),
                                   ann_path=self.get_ann_path(item_name))

    def get_related_images(self, item_name):
        results = []
        path = self.get_related_images_path(item_name)
        if dir_exists(path):
            files = list_files(path, SUPPORTED_IMG_EXTS)
            for file in files:
                img_meta_path = os.path.join(path, get_file_name_with_ext(file)+".json")
                img_meta = {}
                if file_exists(img_meta_path):
                    img_meta = load_json_file(img_meta_path)
                results.append((file, img_meta))
        return results


class PointcloudProject(VideoProject):
    dataset_class = PointcloudDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = PointcloudDataset

    @classmethod
    def read_single(cls, dir):
        return read_project_wrapper(dir, cls)


def download_pointcloud_project(api, project_id, dest_dir, dataset_ids=None, download_items=True, log_progress=False):
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

        ds_progress = None
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

            ds_progress.iters_done_report(len(batch))

    project_fs.set_key_id_map(key_id_map)



