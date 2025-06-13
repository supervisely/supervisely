import os
from collections import defaultdict
from typing import Dict, Union, Optional

from supervisely import ProjectMeta, generate_free_name, is_development, logger
from supervisely.api.api import Api, ApiContext
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.imaging.image import SUPPORTED_IMG_EXTS
from supervisely.io.fs import get_file_ext, list_files
from supervisely.project.project_settings import LabelingInterface


class MultiViewImageConverter(ImageConverter):

    def __init__(
            self,
            input_data: str,
            labeling_interface: Optional[Union[LabelingInterface, str]],
            upload_as_links: bool,
            remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)

        self._supports_links = True
        self._force_shape_for_links = self.upload_as_links

    def __str__(self):
        return AvailableImageConverters.MULTI_VIEW

    def validate_labeling_interface(self) -> bool:
        """Only multi_view labeling interface can be used to group images on single screen."""
        return self._labeling_interface == LabelingInterface.MULTIVIEW

    def validate_format(self) -> bool:
        logger.debug(f"Validating format: {self.__str__()}")
        group_map = self._find_image_directories()
        if not group_map:
            logger.debug(f"Input data does not match {str(self)} format.")
            return False
        else:
            self._group_map = group_map
            logger.debug(f"Found multi-view images in {self._input_data}.")
            return True

    def _find_image_directories(self) -> Union[Dict[str, list], None]:
        group_map = defaultdict(list)
        ann_exts = [".json", ".xml", ".txt"]
        for root, _, files in os.walk(self._input_data):
            for file in files:
                if get_file_ext(file) in ann_exts:
                    return None
                if get_file_ext(file) in SUPPORTED_IMG_EXTS:
                    group_map[root].append(os.path.join(root, file))
            
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
        project_id = dataset.project_id
        api.project.set_multiview_settings(project_id)

        meta_json = api.project.get_meta(project_id, with_settings=True)
        meta = ProjectMeta.from_json(meta_json)

        existing_names = set([info.name for info in api.image.get_list(dataset.id)])
        items_count = sum(len(images) for images in self._group_map.values())
        if log_progress:
            progress, progress_cb = self.get_progress(items_count, "Uploading images...")
        else:
            progress_cb = None

        for group_path, image_group in self._group_map.items():
            logger.info(f"Found files in {group_path}.")

            group_name = os.path.basename(group_path.rstrip("/"))
            images = []
            for image in image_group:
                name = os.path.basename(image)
                new_name = generate_free_name(
                    existing_names, name, with_ext=True, extend_used_names=True
                )
                if new_name != name:
                    logger.warn(f"Image '{name}' already exists. Renamed to '{new_name}'.")
                    os.rename(image, os.path.join(group_path, new_name))
                    image = os.path.join(group_path, new_name)
                if self._upload_as_links:
                    image = self.remote_files_map.get(image, image)
                images.append(image)

            with ApiContext(
                api=api, project_id=project_id, dataset_id=dataset_id, project_meta=meta
            ):
                kwarg = {"links": images} if self.upload_as_links else {"paths": images}
                api.image.upload_multiview_images(
                    dataset.id, group_name, **kwarg, progress_cb=progress_cb
                )

        if log_progress:
            if is_development():
                progress.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")
