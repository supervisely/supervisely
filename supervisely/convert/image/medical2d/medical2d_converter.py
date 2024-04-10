import os
from collections import defaultdict, namedtuple
from typing import Dict, List

import cv2
import nrrd
import numpy as np
import pydicom
import tifffile
from tqdm import tqdm

from supervisely import Api, ProjectMeta, logger
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.imaging.image import is_valid_ext
from supervisely.io.fs import (
    dirs_filter,
    get_file_ext,
    get_file_name_with_ext,
    list_files,
    list_files_recursively,
)

SPLIT_TO_CHANNELS_DIR_NAME = "split"
UPLOAD_AS_IMAGES_DIR_NAME = "images"
ImageGroup = namedtuple("ImageGroup", ["split", "upload"])


class Medical2DImageConverter(ImageConverter):
    allowed_exts = [".nrrd", ".dcm", ".DCM", ".dicom", ".DICOM", ".nii", ".nii.gz"]

    def __init__(self, input_data: str, labeling_interface: str) -> None:
        self._input_data: str = input_data
        self._items: List[ImageConverter.Item] = []
        self._meta: ProjectMeta = None
        self._labeling_interface = labeling_interface
        self._filtered = None

    def __str__(self):
        return AvailableImageConverters.MEDICAL2D

    def validate_labeling_interface(self) -> bool:
        """Only medical labeling interface can be used for medical images."""
        return self._labeling_interface == "medical_imaging_single"

    def validate_format(self) -> bool:
        logger.debug(f"Validating format: {self.__str__()}")

        files = list_files_recursively(self._input_data, valid_extensions=self.allowed_exts)
        self._filtered = self._filter2d(files)

        if len(files) == 0:
            logger.debug(f"No medical images in 2D format were found in {self._input_data!r}.")
            return False
        else:
            logger.debug(f"The medical images in 2D format were found in {self._input_data!r}.")
            return True

    def _filter2d(self, files):
        for i, file in enumerate(files):
            if get_file_ext(file).lower() == ".dcm":
                ds = pydicom.dcmread(file)
                num_frames = ds.get("NumberOfFrames", 1)
                if num_frames > 1:
                    files.pop(i)
        return files

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
    ) -> None:
        """Upload converted data to Supervisely"""
        # TODO: Move this part of the code to some preparation method
        dataset = api.dataset.get_info_by_id(dataset_id)
        if self._meta is not None:
            curr_meta = self._meta
        else:
            curr_meta = ProjectMeta()
        meta_json = api.project.get_meta(dataset.project_id)
        meta = ProjectMeta.from_json(meta_json)
        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        api.project.update_meta(dataset.project_id, meta)

        # TODO: take multiview settings from tags in .dcm
        # api.project.set_multiview_settings(dataset.project_id)

        if log_progress:
            progress = tqdm(total=self.items_count, desc="Uploading images...")
            progress_cb = progress.update
        else:
            progress_cb = None

        names = [get_file_name_with_ext(x) for x in self._filtered]
        api.image.upload_paths(dataset.id, names, self._filtered, progress_cb)

        # for group_path, image_group in self._group_map.items():
        #     logger.info(f"Found files in {group_path}.")

        #     images = []
        #     channels = []

        #     images_to_split, images_to_upload = image_group.split, image_group.upload

        #     group_name = os.path.basename(group_path)
        #     for image_to_upload in images_to_upload:
        #         images.append(os.path.join(group_path, image_to_upload))
        #     for image_to_split in images_to_split:
        #         channels.extend(self._get_image_channels(os.path.join(group_path, image_to_split)))
        #     api.image.upload_multispectral(dataset.id, group_name, channels, images, progress_cb)

        if log_progress:
            progress.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")

    def _get_image_channels(self, file_path: str) -> List[np.ndarray]:
        file_ext = get_file_ext(file_path).lower()
        logger.debug(f"Working with file {file_path} with extension {file_ext}.")

        if file_ext == ".nrrd":
            logger.debug(f"Found nrrd file: {file_path}.")
            image, _ = nrrd.read(file_path)
        elif file_ext == ".dcm":
            pass
        # elif file_ext == ".tif":
        #     logger.debug(f"Found tiff file: {file_path}.")
        #     image = tifffile.imread(file_path)
        # elif is_valid_ext(file_ext):
        #     logger.debug(f"Found image file: {file_path}.")
        #     image = cv2.imread(file_path)
        # else:
        #     logger.warning(f"File {file_path} has unsupported extension.")
        #     return

        return [image[:, :, i] for i in range(image.shape[2])]
