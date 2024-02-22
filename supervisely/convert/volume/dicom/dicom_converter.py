import os
from typing import List

from supervisely import ProjectMeta, VolumeAnnotation, logger
from supervisely.convert.base_converter import AvailableVolumeConverters
from supervisely.convert.volume.volume_converter import VolumeConverter
from supervisely.io.fs import JUNK_FILES, get_file_ext, get_file_name
from supervisely.volume.volume import is_valid_ext as validate_volume_ext


class DICOMConverter(VolumeConverter):
    def __init__(self, input_data: str):
        self._input_data: str = input_data
        self._items: List[VolumeConverter.Item] = []
        self._meta: ProjectMeta = None

    def __str__(self) -> str:
        return AvailableVolumeConverters.DICOM

    @property
    def ann_ext(self) -> str:
        return None

    @property
    def key_file_ext(self) -> str:
        return None

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta) -> bool:
        return False  # remove this method?

    def validate_key_file(self, key_file_path: str) -> bool:
        return False  # remove this method?

    def validate_format(self) -> bool:
        detected_ann_cnt = 0
        vol_list = [], {}, {}
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if ext == ".dcm":
                    vol_list.append(full_path)
                    detected_ann_cnt += 1
                else:
                    # need check files without extension
                    # or use api.volume
                    pass

        meta = ProjectMeta()

        # create Items
        self._items = []
        for vol_path in vol_list:
            item = self.Item(vol_path)
            self._items.append(item)
        self._meta = meta
        return detected_ann_cnt > 0

    def to_supervisely(
        self, item: VolumeConverter.Item, meta: ProjectMeta = None
    ) -> VolumeAnnotation:
        """Convert to Supervisely format."""
        return item.create_empty_annotation()
