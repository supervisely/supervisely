from typing import List

import supervisely.convert.volume.dicom.dicom_helper as dicom_helper
from supervisely import ProjectMeta, VolumeAnnotation
from supervisely.convert.base_converter import AvailableVolumeConverters
from supervisely.convert.volume.volume_converter import VolumeConverter
from supervisely.volume.volume import inspect_dicom_series


class DICOMConverter(VolumeConverter):
    def __init__(self, input_data: str, labeling_interface: str):
        self._input_data: str = input_data
        self._items: List[VolumeConverter.Item] = []
        self._meta: ProjectMeta = None
        self._labeling_interface: str = labeling_interface

    def __str__(self) -> str:
        return AvailableVolumeConverters.DICOM

    @property
    def ann_ext(self) -> str:
        return None

    @property
    def key_file_ext(self) -> str:
        return None

    def validate_format(self) -> bool:
        # DICOM
        series_infos = inspect_dicom_series(root_dir=self._input_data)

        # create Items
        self._items = []
        for dicom_id, dicom_paths in series_infos.items():
            item_path, volume_meta = dicom_helper.dcm_to_nrrd(dicom_id, dicom_paths)
            item = self.Item(item_path, volume_meta=volume_meta)
            self._items.append(item)
        self._meta = ProjectMeta()

        return len(series_infos) > 0

    def to_supervisely(
        self,
        item: VolumeConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> VolumeAnnotation:
        """Convert to Supervisely format."""
        return None
