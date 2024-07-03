import os
from collections import defaultdict, namedtuple
from typing import Dict, List

import cv2
import nrrd
import numpy as np

from supervisely import is_development, logger, ProjectMeta
from supervisely.api.api import Api, ApiContext
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.image.image_helper import read_tiff_image
from supervisely.imaging.image import is_valid_ext
from supervisely.io.fs import dirs_filter, get_file_ext, list_files
from supervisely.project.project_settings import LabelingInterface

SPLIT_TO_CHANNELS_DIR_NAME = "split"
UPLOAD_AS_IMAGES_DIR_NAME = "images"
ImageGroup = namedtuple("ImageGroup", ["split", "upload"])


class MultiSpectralImageConverter(ImageConverter):

    def __str__(self):
        return AvailableImageConverters.MULTISPECTRAL

    def validate_labeling_interface(self) -> bool:
        """Only multispectral labeling interface can be used for multispectral images."""
        return self._labeling_interface == LabelingInterface.MULTISPECTRAL

    def validate_format(self) -> bool:
        logger.debug(f"Validating format: {self.__str__()}")
        group_map = self._find_image_directories()
        if not group_map:
            logger.debug(f"No multispectral images found in {self._input_data}.")
            return False
        else:
            self._group_map = group_map
            logger.debug(f"Found multispectral images in {self._input_data}.")
            return True

    def _find_image_directories(self) -> Dict[str, ImageGroup]:
        group_map = defaultdict(ImageGroup)
        for checked_directory in dirs_filter(self._input_data, self._check_directory):
            split_images = []
            upload_images = []

            split_dir = os.path.join(checked_directory, SPLIT_TO_CHANNELS_DIR_NAME)

            images_dir = os.path.join(checked_directory, UPLOAD_AS_IMAGES_DIR_NAME)

            if os.path.isdir(split_dir):
                split_images = list_files(split_dir)
            if os.path.isdir(images_dir):
                upload_images = list_files(images_dir)

            group_map[checked_directory] = ImageGroup(split_images, upload_images)

        return group_map

    def _check_directory(self, path: str) -> bool:
        split_dir = os.path.join(path, SPLIT_TO_CHANNELS_DIR_NAME)
        images_dir = os.path.join(path, UPLOAD_AS_IMAGES_DIR_NAME)

        if os.path.isdir(split_dir) or os.path.isdir(images_dir):
            return True
        return False

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
    ) -> None:
        """Upload converted data to Supervisely"""
        dataset = api.dataset.get_info_by_id(dataset_id)
        project_id = dataset.project_id
        api.project.set_multispectral_settings(project_id)

        meta_json = api.project.get_meta(project_id, with_settings=True)
        meta = ProjectMeta.from_json(meta_json)

        items_count = sum(len(group.split) + len(group.upload) for group in self._group_map.values())
        if log_progress:
            progress, progress_cb = self.get_progress(items_count, "Uploading images...")
        else:
            progress_cb = None

        for group_path, image_group in self._group_map.items():
            logger.info(f"Found files in {group_path}.")

            images = []
            channels = []

            images_to_split, images_to_upload = image_group.split, image_group.upload

            group_name = os.path.basename(group_path)
            for image_to_upload in images_to_upload:
                images.append(os.path.join(group_path, image_to_upload))
            for image_to_split in images_to_split:
                channels.extend(self._get_image_channels(os.path.join(group_path, image_to_split)))

            with ApiContext(
                api=api, project_id=project_id, dataset_id=dataset_id, project_meta=meta
            ):
                api.image.upload_multispectral(dataset.id, group_name, channels, images, progress_cb)

        if log_progress:
            if is_development():
                progress.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")

    def _get_image_channels(self, file_path: str) -> List[np.ndarray]:
        file_ext = get_file_ext(file_path).lower()
        logger.debug(f"Working with file {file_path} with extension {file_ext}.")

        if file_ext == ".nrrd":
            logger.debug(f"Found nrrd file: {file_path}.")
            image, _ = nrrd.read(file_path)
        elif file_ext in [".tif", ".tiff"]:
            image = read_tiff_image(file_path)
        elif is_valid_ext(file_ext):
            logger.debug(f"Found image file: {file_path}.")
            image = cv2.imread(file_path)
        else:
            logger.warning(f"File {file_path} has unsupported extension.")
            return
        
        if image is None:
            logger.warning(f"Failed to read image {file_path}.")
            return

        return [image[:, :, i] for i in range(image.shape[2])]
