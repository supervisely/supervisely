import os
from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm

from supervisely import Api, logger, ProjectMeta
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.imaging.image import SUPPORTED_IMG_EXTS
from supervisely.io.fs import get_file_ext, list_files

DEFAULT_GROUP_NAME = "group"


class MultiViewImageConverter(ImageConverter):
    def __init__(self, input_data: str, labeling_interface: str) -> None:
        self._input_data: str = input_data
        self._items: List[ImageConverter.Item] = []
        self._meta: ProjectMeta = None
        self._labeling_interface = labeling_interface

    def __str__(self):
        return AvailableImageConverters.MULTI_VIEW

    def validate_labeling_interface(self) -> bool:
        """Only multi_view labeling interface can be used to group images on single screen."""
        return self._labeling_interface == "multi_view"

    def validate_format(self) -> bool:
        logger.debug(f"Validating format: {self.__str__()}")
        group_map = self._find_image_directories()
        if not group_map:
            logger.debug(f"No multi-view images found in {self._input_data}.")
            return False
        else:
            self._group_map = group_map
            logger.debug(f"Found multi-view images in {self._input_data}.")
            return True

    def _find_image_directories(self) -> Dict[str, list]:
        group_map = defaultdict(list)
        for root, _, files in os.walk(self._input_data):
            if any([get_file_ext(file) in SUPPORTED_IMG_EXTS for file in files]):
                group_map[root] = list_files(root, SUPPORTED_IMG_EXTS)
        return group_map

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
    ) -> None:
        """Upload converted data to Supervisely"""
        dataset = api.dataset.get_info_by_id(dataset_id)
        api.project.set_multiview_settings(dataset.project_id)

        existing_names = [info.name for info in api.image.get_list(dataset.id)]
        items_count = sum(len(images) for images in self._group_map.values())
        if log_progress:
            progress = tqdm(total=items_count, desc="Uploading images...")
            progress_cb = progress.update
        else:
            progress_cb = None

        for group_path, image_group in self._group_map.items():
            logger.info(f"Found files in {group_path}.")

            group_name = os.path.basename(group_path.rstrip("/"))
            images = [os.path.join(group_path, image) for image in image_group]
            checked_images = []
            for image in images:
                if os.path.basename(image) in existing_names:
                    name, ext = os.path.splitext(os.path.basename(image))
                    i = 1
                    new_name = f"{os.path.basename(name)}_{i}{ext}"
                    while new_name in existing_names:
                        new_name = f"{os.path.basename(name)}_{i}{ext}"
                        i += 1
                    logger.warn(f"Image '{image}' already exists. Renamed to '{new_name}'.")
                    os.rename(image, os.path.join(group_path, new_name))
                    image = os.path.join(group_path, new_name)
                    existing_names.append(new_name)
                else:
                    existing_names.append(os.path.basename(image))
                checked_images.append(image)
            images = checked_images

            api.image.upload_multiview_images(
                dataset.id, group_name, images, progress_cb=progress_cb
            )

        if log_progress:
            progress.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")
