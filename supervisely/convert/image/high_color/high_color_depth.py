import os
from collections import defaultdict
from typing import Dict, Generator, Union

import cv2
import numpy as np
import tqdm

from supervisely import ProjectMeta, generate_free_name, is_development, logger
from supervisely._utils import batched
from supervisely.annotation.annotation import Annotation
from supervisely.api.api import Api
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.high_color import high_color_helper as helpers
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.env import task_id as get_task_id
from supervisely.io.env import team_id as get_team_id
from supervisely.io.fs import list_files_recursively
from supervisely.io.json import load_json_file
from supervisely.project.project_settings import LabelingInterface
from supervisely.team_files import RECOMMENDED_IMPORT_BACKUP_PATH


class HighColorDepthImageConverter(ImageConverter):
    # allowed_exts = [".png", ".tiff", ".tif", ".bmp", ".exr", ".hdr"]
    allowed_exts = [".exr", ".hdr"]

    class Item(ImageConverter.Item):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._original_path = None

        @property
        def original_path(self) -> Union[str, None]:
            return self._original_path

        @original_path.setter
        def original_path(self, original_path: str) -> None:
            self._original_path = original_path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._team_id = None

    def __str__(self):
        return AvailableImageConverters.HIGH_COLOR_DEPTH

    def validate_labeling_interface(self) -> bool:
        """Only multi_view labeling interface can be used to group images on single screen."""
        return self._labeling_interface in [
            LabelingInterface.DEFAULT,
            LabelingInterface.IMAGES_WITH_16_COLOR,
        ]

    def validate_format(self) -> bool:
        logger.debug(f"Validating format: {self.__str__()}")

        for image_path in self._find_high_color_depth_image_generator(self._input_data):
            item = self.Item(image_path)
            self._items.append(item)

        if self.items_count == 0:
            logger.debug(f"Not found any images with high color depth in '{self._input_data}'")
            return False

        return True

    def _find_high_color_depth_image_generator(self, input_data: str) -> Generator[str, None, None]:
        for image_path in list_files_recursively(input_data, valid_extensions=self.allowed_exts):
            if self._is_high_color_depth_image(image_path):
                yield image_path

    def _is_high_color_depth_image(self, image_path: str) -> bool:
        image = helpers.read_high_color_images(image_path)
        if image is None:
            return False
        return image.dtype in self._supported_depths

    @property
    def _supported_depths(self):
        return [np.uint16, np.int16, np.float16, np.float32, np.float64, np.uint32]

    def to_supervisely(self, *args, **kwargs):
        return None

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
    ) -> None:
        """Upload converted data to Supervisely"""
        team_id = get_team_id()
        backup_dir = os.path.join(RECOMMENDED_IMPORT_BACKUP_PATH, str(get_task_id()))

        # progress, progress_cb = None, None
        # if log_progress:
        #     progress, progress_cb = self.get_progress(
        #         len(self.get_items()),
        #         "Backing up original files and converting images to nrrd...",
        #     )

        # Back up original files to avoid data loss and convert images to nrrd format
        with tqdm.tqdm(
            total=len(self.get_items()),
            desc="Backing up original files and converting images to nrrd...",
        ) as pbar:
            for batch_items in batched(self.get_items(), batch_size=batch_size):

                local_paths, remote_paths = [], []

                for item in batch_items:
                    local_paths.append(item.path)

                    remote_path = os.path.join(backup_dir, item.name)
                    remote_paths.append(remote_path)

                    item.original_path = remote_path
                    image = helpers.read_high_color_images(item.path)

                    orig_item_name = item.name
                    nrrd_path = item.path + ".nrrd"
                    nrrd_path = helpers.save_nrrd(image, nrrd_path)
                    item.path = nrrd_path

                    item_meta = {}
                    # Add original file path to image meta
                    item_meta["original_file_path"] = os.path.join(backup_dir, orig_item_name)
                    if item.meta:
                        item_meta.update(item.meta)
                    item.set_meta_data(item_meta)

                api.file.upload_bulk(team_id, local_paths, remote_paths)
                pbar.update(len(batch_items))

        if log_progress and is_development():
            pbar.close()

        super().upload_dataset(api, dataset_id, batch_size, log_progress, self.get_items())
