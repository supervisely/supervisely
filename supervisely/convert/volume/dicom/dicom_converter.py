import os
from pathlib import Path
from typing import List

from supervisely import ProjectMeta, generate_free_name, logger
from supervisely.api.api import Api
from supervisely.convert.base_converter import AvailableVolumeConverters
from supervisely.convert.volume.dicom import dicom_helper as h
from supervisely.convert.volume.volume_converter import VolumeConverter
from supervisely.io.fs import get_file_ext, get_file_name_with_ext
from supervisely.volume.volume import (
    get_extension,
    inspect_dicom_series,
    read_dicom_serie_volume,
)

# Volume files uploaded as they are when found next to DICOM series.
_STANDALONE_VOLUME_EXTS = [".nrrd"]

# How many skipped files the warning names; the count always covers all of them.
_MAX_SKIPPED_FILES_IN_LOG = 50


class DICOMConverter(VolumeConverter):
    """Imports DICOM series (CT/MRI slices) into Supervisely volume project; groups by series UID.

    NRRD volumes found in the same input are imported as they are, after the series, so a folder
    that mixes DICOM series and NRRD files is imported whole. Other files are reported as skipped.
    """

    class Item(VolumeConverter.Item):
        """Item class for DICOM series."""

        def __init__(self, serie_id: str, item_paths: List[str], volume_meta: dict):
            """
            :param serie_id: Series UID from DICOM.
            :type serie_id: str
            :param item_paths: Paths to DICOM slices.
            :type item_paths: List[str]
            :param volume_meta: Volume metadata (spacing, origin, etc.).
            :type volume_meta: dict
            """
            item_path = item_paths[0] if len(item_paths) > 0 else None
            super().__init__(item_path, volume_meta=volume_meta)

            self._serie_id: str = serie_id
            self._item_paths: List[str] = item_paths

        @property
        def item_paths(self) -> List[str]:
            return self._item_paths

        @item_paths.setter
        def item_paths(self, paths: List[str]) -> None:
            self._item_paths = paths

        @property
        def serie_id(self) -> str:
            return self._serie_id

        @serie_id.setter
        def serie_id(self, serie_id: str) -> None:
            self._serie_id = serie_id

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
        used_paths = set()
        for dicom_id, dicom_paths in series_infos.items():
            if len(dicom_paths) == 0:
                logger.warning(f"Empty serie {dicom_id}, serie will be skipped")
                continue
            item_path = dicom_paths[0]
            if get_extension(path=item_path) is None:
                logger.warning(
                    f"Can not recognize file extension {item_path}, serie will be skipped"
                )
                continue

            for dicom_path in dicom_paths:
                h.read_and_convert_to_monochrome2(dicom_path)
            _, meta = read_dicom_serie_volume(dicom_paths, anonymize=True)
            item = self.Item(
                serie_id=dicom_id, item_paths=dicom_paths, volume_meta=meta
            )
            self._items.append(item)
            used_paths.update(os.path.normpath(p) for p in dicom_paths)
        self._meta = ProjectMeta()

        if self.items_count == 0:
            # no series: NRRD-only input is left to the other volume converters
            return False

        # NRRD next to the series: imported as they are, like the "Import DICOM Volumes" app
        for path in self._list_standalone_volumes(used_paths):
            try:
                self._items.append(VolumeConverter.Item(path))
                used_paths.add(path)
            except Exception as e:
                logger.warning(
                    f"Failed to read volume {get_file_name_with_ext(path)}, it will be skipped: "
                    f"{repr(e)}"
                )

        self._report_skipped_files(used_paths)
        return True

    def _list_standalone_volumes(self, used_paths: set) -> List[str]:
        paths = []
        for root, _, files in os.walk(self._input_data):
            for file in files:
                path = os.path.normpath(os.path.join(root, file))
                if path in used_paths:
                    continue
                if get_file_ext(path).lower() in _STANDALONE_VOLUME_EXTS:
                    paths.append(path)
        return sorted(paths)

    def _report_skipped_files(self, used_paths: set) -> List[str]:
        skipped = []
        for root, _, files in os.walk(self._input_data):
            for file in files:
                path = os.path.normpath(os.path.join(root, file))
                if path not in used_paths:
                    skipped.append(os.path.relpath(path, self._input_data))
        if skipped:
            skipped.sort()
            names = ", ".join(skipped[:_MAX_SKIPPED_FILES_IN_LOG])
            if len(skipped) > _MAX_SKIPPED_FILES_IN_LOG:
                names += f" and {len(skipped) - _MAX_SKIPPED_FILES_IN_LOG} more"
            logger.warning(
                f"{len(skipped)} file(s) are neither DICOM series slices nor NRRD volumes "
                f"and will not be imported: {names}"
            )
        return skipped

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 1,
        log_progress=True,
    ):
        existing_names = set([vol.name for vol in api.volume.get_list(dataset_id)])

        dicom_items = [item for item in self._items if isinstance(item, self.Item)]
        volume_items = [item for item in self._items if not isinstance(item, self.Item)]

        for item in dicom_items:
            serie_id, serie_paths = item.serie_id, item.item_paths
            volume_name = f"{serie_id}.nrrd"
            volume_name = generate_free_name(
                existing_names, volume_name, with_ext=True, extend_used_names=True
            )
            api.volume.upload_dicom_serie_paths(
                dataset_id=dataset_id,
                name=volume_name,
                paths=serie_paths,
                log_progress=log_progress,
                anonymize=True,
            )

        for item in volume_items:
            # same naming as the NRRD-only import (VolumeConverter.upload_dataset)
            ext = get_file_ext(item.path)
            if ext.lower() != ext:
                new_path = Path(item.path).with_suffix(ext.lower()).as_posix()
                os.rename(item.path, new_path)
                item.path = new_path
            volume_name = generate_free_name(
                existing_names,
                get_file_name_with_ext(item.path),
                with_ext=True,
                extend_used_names=True,
            )
            item.name = volume_name
            api.volume.upload_nrrd_series_paths(
                dataset_id, [volume_name], [item.path], log_progress=log_progress
            )

        if volume_items:
            logger.info(
                f"Uploaded {len(dicom_items)} DICOM series and {len(volume_items)} NRRD volume(s)."
            )
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")
