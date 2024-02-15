import imghdr
import os
from typing import Literal

from supervisely import Api, ProjectType, batched, logger
from supervisely.convert.base_converter import BaseConverter
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.video.video_converter import VideoConverter
from supervisely.imaging.image import is_valid_ext as is_valid_image_ext
from supervisely.io.fs import get_file_ext
from supervisely.pointcloud.pointcloud import is_valid_ext as is_valid_point_cloud_ext
from supervisely.video.video import is_valid_ext as is_valid_video_ext
from supervisely.volume.volume import is_valid_ext as is_valid_volume_ext

possible_annotations_exts = [".json", ".xml", ".txt"]
possible_junk_exts = [".DS_Store"]

# class Converter:
class ImportManager:
    def __init__(
        self,
        input_data
    ):
        # input_data date - folder / archive / link / team files
        # if save_path is None - save to the same level folder
        if not os.path.exists(input_data):
            raise RuntimeError(f"Directory does not exist: {input_data}")

        self.input_data = input_data
        self.items = {}
        self.annotations = {}

        self.modality = self._detect_modality()
        self.converter = self._get_converter()
        self.api = Api.from_env()

    def _detect_modality(self):
        """Detect modality of input data (images, videos, pointclouds, volumes)"""

        if self._contains_only_images():
            return ProjectType.IMAGES.value

        if self._contains_only_videos():
            return ProjectType.VIDEOS.value

        if self._contains_only_point_clouds():
            return ProjectType.POINT_CLOUDS.value  # @TODO: ProjectType.POINT_CLOUDS_EPISODES

        if self._contains_only_volumes():
            return ProjectType.VOLUMES.value
        else:
            raise RuntimeError("Use of mixed data types is not supported.")

    def _get_converter(self):
        """Return correct converter"""
        if self.modality == ProjectType.IMAGES.value:
            return ImageConverter(self.input_data, self.items, self.annotations).converter
        elif self.modality == ProjectType.VIDEOS.value:
            return VideoConverter(self.input_data).converter
        # elif self.modality == ProjectType.POINT_CLOUDS.value:
        #     return PointCloudConverter(input_data)
        # elif self.modality == ProjectType.VOLUMES.value:
        #     return VolumeConverter(input_data)

    def upload_dataset(self, dataset_id):
        """Upload converted data to Supervisely"""
        meta = self.converter.get_meta()
        items = self.converter.get_items()
        for batch in batched(items, batch_size=50):
            item_names = list(batch.keys())
            img_paths = [item["image"] for item in batch]
            ann_paths = [item["ann"] for item in batch]

            anns = []
            for img_path, ann_path in zip(img_paths, ann_paths):
                ann = self.converter.to_supervisely(img_path, ann_path, meta)
                anns.append(ann)

            img_infos = self.api.image.upload_paths(dataset_id, item_names, img_paths)
            img_ids = [img_info.id for img_info in img_infos]
            self.api.annotation.upload_anns(img_ids, anns)


    def _contains_only_images(self):
        for root, _, files in os.walk(self.input_data):
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if ext in possible_junk_exts:  # add better check
                    continue
                elif ext in possible_annotations_exts:
                    self.annotations[full_path] = None
                elif imghdr.what(full_path) is None:
                    logger.info(f"Non-image file found: {full_path}")
                    return False
                else:
                    self.items[full_path] = None

        return True


    def _contains_only_videos(self):
        for root, _, files in os.walk(self.input_data):
            for file in files:
                ext = get_file_ext(file)
                if is_valid_video_ext(ext):
                    logger.info(f"Non-video file found: {os.path.join(root, file)}")
                    return False
        return True


    def _contains_only_point_clouds(self):
        for root, _, files in os.walk(self.input_data):
            for file in files:
                ext = get_file_ext(file)
                if is_valid_point_cloud_ext(ext):
                    logger.info(f"Non-point cloud file found: {os.path.join(root, file)}")
                    return False
        return True


    def _contains_only_volumes(self):
        for root, _, files in os.walk(self.input_data):
            for file in files:
                ext = get_file_ext(file)
                if is_valid_volume_ext(ext):
                    logger.info(f"Non-volume file found: {os.path.join(root, file)}")
                    return False
        return True