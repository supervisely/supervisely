import os
from collections import defaultdict, namedtuple
from typing import Dict, List, Optional

import cv2
import nrrd
import numpy as np

from supervisely import ProjectMeta, is_development, logger
from supervisely._utils import generate_free_name
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.tag import Tag
from supervisely.api.api import Api, ApiContext
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.image.image_helper import read_tiff_image
from supervisely.imaging.image import is_valid_ext
from supervisely.io.fs import dirs_filter, get_file_ext, list_files
from supervisely.project.project_settings import LabelingInterface
from supervisely.project.project_type import _MULTISPECTRAL_TAG_NAME

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
        group_tag_meta = meta.get_tag_meta(_MULTISPECTRAL_TAG_NAME)

        items_count = sum(
            len(group.split) + len(group.upload) for group in self._group_map.values()
        )
        if log_progress:
            progress, progress_cb = self.get_progress(items_count, "Uploading images...")
        else:
            progress_cb = None

        for group_path, image_group in self._group_map.items():
            logger.info(f"Found files in {group_path}.")

            images, channels = [], []
            nrrd_paths, nrrds_anns, nrrd_names = [], [], []

            images_to_split, images_to_upload = image_group.split, image_group.upload

            group_name = os.path.basename(group_path)
            for image_to_upload in images_to_upload:
                images.append(os.path.join(group_path, image_to_upload))
            for image_to_split in images_to_split:
                image = self._get_image(os.path.join(group_path, image_to_split))
                if image is None:
                    continue

                # image has wide range of pixel values, we will upload it as nrrd
                # (supports `windowing` feature in the Image Labeling Toolbox)
                img_channels = self._get_image_channels(image)
                if image.dtype != np.uint8 and np.max(image) > 255:
                    for idx, channel in enumerate(img_channels):
                        nrrd_path = self._prepare_nrrd(channel, group_path, image_to_split, idx)
                        nrrd_paths.append(nrrd_path)
                        nrrd_names.append(os.path.basename(nrrd_path))
                        group_tag = Tag(meta=group_tag_meta, value=group_name)
                        nrrds_anns.append(Annotation(channel.shape).add_tag(group_tag))
                else:
                    channels.extend(img_channels)

            with ApiContext(
                api=api, project_id=project_id, dataset_id=dataset_id, project_meta=meta
            ):
                if channels or images:
                    api.image.upload_multispectral(
                        dataset.id, group_name, channels, images, progress_cb
                    )
                if nrrd_paths:
                    img_infos = api.image.upload_paths(
                        dataset.id, nrrd_names, nrrd_paths, progress_cb
                    )
                    image_ids = [image_info.id for image_info in img_infos]
                    api.annotation.upload_anns(image_ids, nrrds_anns)

        if log_progress:
            if is_development():
                progress.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")

    def _get_image(self, file_path: str) -> Optional[np.ndarray]:
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

        return image

    def _get_image_channels(self, image) -> List[np.ndarray]:
        if image is None:
            return []

        # Handle single-channel images (grayscale)
        if len(image.shape) == 2:
            logger.debug(f"Image has single channel.")
            return [image]

        # Handle multi-channel images
        return [image[:, :, i] for i in range(image.shape[2])]

    def _prepare_nrrd(
        self, image: np.ndarray, group_path: str, image_name: str, channel_idx: int
    ) -> str:
        header = {"sizes": image.shape, "type": str(image.dtype), "dimension": len(image.shape)}

        name = os.path.splitext(image_name)[0] + f"_{channel_idx}.nrrd"
        path = os.path.join(group_path, name)
        if os.path.exists(path):
            path = generate_free_name(os.listdir(group_path), path, with_ext=True)

        nrrd.write(path, image, header)
        return path
