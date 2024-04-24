import os
from typing import List, Union

import magic
import nrrd
import numpy as np

from supervisely import Annotation, ProjectMeta, logger
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.image.medical2d import medical2d_helper as helper
from supervisely.io.fs import remove_dir, get_file_ext, mkdir
from supervisely.volume.volume import is_nifti_file


# @TODO: add group tags?
class Medical2DImageConverter(ImageConverter):

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
        return self._labeling_interface in ["default", "medical_imaging_single", "images_with_16_color"]

    def validate_format(self) -> bool:
        logger.debug(f"Validating format: {self.__str__()}")

        converted_dir_name = "converted"
        converted_dir = os.path.join(self._input_data, converted_dir_name)
        mkdir(converted_dir, remove_content_if_exists=True)

        nrrd = {}
        for root, _, files in os.walk(self._input_data):
            if converted_dir_name in root:
                continue
            for file in files:
                path = os.path.join(root, file)
                ext = get_file_ext(path).lower()
                mime = magic.from_file(path, mime=True)
                if mime == "application/dicom":
                    if helper.is_dicom_file(path):  # is dicom
                        paths, names = helper.convert_dcm_to_nrrd(path, converted_dir)
                        for path, name in zip(paths, names):
                            nrrd[name] = path
                elif ext == ".nrrd":
                    if helper.check_nrrd(path):  # is nrrd
                        paths, names = helper.slice_nrrd_file(path, converted_dir)
                        for path, name in zip(paths, names):
                            nrrd[name] = path
                elif mime == "application/gzip" or mime == "application/octet-stream":
                    if is_nifti_file(path):  # is nifti
                        paths, names = helper.slice_nifti_file(path, converted_dir)
                        for path, name in zip(paths, names):
                            nrrd[name] = path

        self._items = []
        for name, path in nrrd.items():
            item = self.Item(item_path=path)
            self._items.append(item)

        if self.items_count == 0:
            remove_dir(converted_dir)

        return self.items_count > 0

    def _get_image_channels(self, file_path: str) -> List[np.ndarray]:
        file_ext = get_file_ext(file_path).lower()
        logger.debug(f"Working with file {file_path} with extension {file_ext}.")

        if file_ext == ".nrrd":
            logger.debug(f"Found nrrd file: {file_path}.")
            image, _ = nrrd.read(file_path)
        elif file_ext == ".dcm":
            pass
        return [image[:, :, i] for i in range(image.shape[2])]

    def to_supervisely(self, item, meta, *args) -> Union[Annotation, None]:
        return None
