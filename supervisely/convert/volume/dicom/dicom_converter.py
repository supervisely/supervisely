from typing import List

from supervisely.api.api import Api
from supervisely import generate_free_name, logger, ProjectMeta
from supervisely.convert.base_converter import AvailableVolumeConverters
from supervisely.convert.volume.volume_converter import VolumeConverter
from supervisely.convert.volume.dicom import dicom_helper as h
from supervisely.volume.volume import inspect_dicom_series, get_extension, read_dicom_serie_volume

class DICOMConverter(VolumeConverter):
    class Item(VolumeConverter.Item):
        """Item class for DICOM series."""
        def __init__(self, serie_id: str, item_paths: List[str], volume_meta: dict):
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
        for dicom_id, dicom_paths in series_infos.items():
            if len(dicom_paths) == 0:
                logger.warn(f"Empty serie {dicom_id}, serie will be skipped")
                continue
            item_path = dicom_paths[0]
            if get_extension(path=item_path) is None:
                logger.warn(
                    f"Can not recognize file extension {item_path}, serie will be skipped"
                )
                continue

            for dicom_path in dicom_paths:
                h.convert_to_monochrome2(dicom_path)
            _, meta = read_dicom_serie_volume(dicom_paths, anonymize=True)
            item = self.Item(serie_id=dicom_id, item_paths=dicom_paths, volume_meta=meta)
            self._items.append(item)
        self._meta = ProjectMeta()

        return self.items_count > 0

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 1,
        log_progress=True,
    ):
        existing_names = set([vol.name for vol in api.volume.get_list(dataset_id)])

        for item in self._items:
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

        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")
