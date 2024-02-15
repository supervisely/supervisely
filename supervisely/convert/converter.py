import os
from typing import Literal

from supervisely import Api, ProjectType, batched
from supervisely.convert.base_converter import BaseConverter
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.modality_helpers import (
    contains_only_images,
    contains_only_point_clouds,
    contains_only_videos,
    contains_only_volumes,
)
from supervisely.convert.video.video_converter import VideoConverter


# class Converter:
class ImportManager:
    def __init__(
        self,
        input_data,
        save_path: str = None,
        output_type: Literal["folder", "archive"] = "folder",
    ):
        # input_data date - folder / archive / link / team files
        # if save_path is None - save to the same level folder
        if not os.path.exists(input_data):
            raise RuntimeError(f"Directory does not exist: {input_data}")

        self.input_data = input_data
        self.save_path = save_path
        self.output_type = output_type

        self.modality = self._detect_modality()
        self.converter = self._get_converter()
        self.api = Api.from_env()

    def _detect_modality(self):
        """Detect modality of input data (images, videos, pointclouds, volumes)"""

        if contains_only_images(self.input_data):
            return ProjectType.IMAGES.value

        if contains_only_videos(self.input_data):
            return ProjectType.VIDEOS.value

        if contains_only_point_clouds(self.input_data):
            return ProjectType.POINT_CLOUDS.value  # @TODO: ProjectType.POINT_CLOUDS_EPISODES

        if contains_only_volumes(self.input_data):
            return ProjectType.VOLUMES.value
        else:
            raise RuntimeError("Use of mixed data types is not supported.")

    def _get_converter(self):
        """Return correct converter"""
        if self.modality == ProjectType.IMAGES.value:
            return ImageConverter(self.input_data).converter
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
