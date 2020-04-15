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
from supervisely_lib.video import video as sly_video

from supervisely_lib.project.project import Dataset, Project, OpenMode
from supervisely_lib.project.project import read_single_project as read_project_wrapper
from supervisely_lib.video_annotation.video_annotation import VideoAnnotation


VideoItemPaths = namedtuple('VideoItemPaths', ['video_path', 'ann_path'])


class VideoDataset(Dataset):
    item_dir_name = 'video'
    annotation_class = VideoAnnotation

    @staticmethod
    def _has_valid_ext(path: str) -> bool:
        return sly_video.has_valid_ext(path)

    def _get_empty_annotaion(self, item_name):
        img_size, frames_count = sly_video.get_image_size_and_frames_count(item_name)
        return self.annotation_class(img_size, frames_count)

    def add_item_np(self, item_name, img, ann=None):
        raise RuntimeError("Deprecated method. Works only with images project")

    def _add_img_np(self, item_name, img):
        raise RuntimeError("Deprecated method. Works only with images project")

    @staticmethod
    def _validate_added_item_or_die(item_path):
        # Make sure we actually received a valid image file, clean it up and fail if not so.
        try:
            sly_video.validate_format(item_path)
        except (sly_video.UnsupportedVideoFormat, sly_video.VideoReadException):
            os.remove(item_path)
            raise

    def set_ann(self, item_name: str, ann):
        if type(ann) is not self.annotation_class:
            raise TypeError("Type of 'ann' have to be Annotation, not a {}".format(type(ann)))
        dst_ann_path = self.get_ann_path(item_name)
        dump_json_file(ann.to_json(), dst_ann_path)

    def get_item_paths(self, item_name) -> VideoItemPaths:
        return VideoItemPaths(video_path=self.get_img_path(item_name), ann_path=self.get_ann_path(item_name))


class VideoProject(Project):
    dataset_class = VideoDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = VideoDataset

    def __init__(self, directory, mode: OpenMode):
        self._key_id_map: KeyIdMap = None
        super().__init__(directory, mode)

    def _read(self):
        super(VideoProject, self)._read()
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


def download_video_project(api, project_id, dest_dir, dataset_ids=None, download_videos=True, log_progress=False):
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
            ds_progress = Progress('Downloading dataset: {!r}'.format(dataset.name), total_cnt=len(videos))
        for batch in batched(videos, batch_size=LOG_BATCH_SIZE):
            video_ids = [video_info.id for video_info in batch]
            video_names = [video_info.name for video_info in batch]

            ann_jsons = api.video.annotation.download_bulk(dataset.id, video_ids)

            for video_id, video_name, ann_json in zip(video_ids, video_names, ann_jsons):
                if video_name != ann_json[ApiField.VIDEO_NAME]:
                    raise RuntimeError("Error in api.video.annotation.download_batch: broken order")

                video_file_path = dataset_fs.generate_item_path(video_name)
                if download_videos is True:
                    api.video.download_path(video_id, video_file_path)
                else:
                    touch(video_file_path)

                dataset_fs.add_item_file(video_name,
                                         video_file_path,
                                         ann=VideoAnnotation.from_json(ann_json, project_fs.meta, key_id_map),
                                         _validate_item=False)

            ds_progress.iters_done_report(len(batch))

    project_fs.set_key_id_map(key_id_map)



