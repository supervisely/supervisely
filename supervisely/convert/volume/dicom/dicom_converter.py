from typing import List

from tqdm import tqdm

import supervisely.convert.volume.dicom.dicom_helper as dicom_helper
from supervisely import Api, generate_free_name, logger, ProjectMeta, VolumeAnnotation
from supervisely.convert.base_converter import AvailableVolumeConverters
from supervisely.convert.volume.volume_converter import VolumeConverter
from supervisely.io.fs import file_exists, silent_remove
from supervisely.io.json import dump_json_file
from supervisely.volume.volume import inspect_dicom_series, inspect_nrrd_series


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
        # NRRD
        # nrrd_paths = inspect_nrrd_series(root_dir=self._input_data)

        # create Items
        self._items = []
        # for path in nrrd_paths:
        #     item = self.Item(path)
        #     self._items.append(item)
        for dicom_id, dicom_paths in series_infos.items():
            item_path, volume_meta = dicom_helper.dcm_to_nrrd(dicom_id, dicom_paths)
            item = self.Item(item_path, volume_meta=volume_meta)
            self._items.append(item)
        self._meta = ProjectMeta()

        return len(series_infos) > 0

    def to_supervisely(
        self, item: VolumeConverter.Item, meta: ProjectMeta = None
    ) -> VolumeAnnotation:
        """Convert to Supervisely format."""
        return item.create_empty_annotation()

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        log_progress=True,
    ):
        """Upload converted data to Supervisely"""

        dataset = api.dataset.get_info_by_id(dataset_id)
        existing_names = set([vol.name for vol in api.volume.get_list(dataset.id)])
        if self._meta is not None:
            curr_meta = self._meta
        else:
            curr_meta = ProjectMeta()
        meta_json = api.project.get_meta(dataset.project_id)
        meta = ProjectMeta.from_json(meta_json)
        meta = meta.merge(curr_meta)

        api.project.update_meta(dataset.project_id, meta)

        if log_progress:
            progress = tqdm(total=self.items_count, desc=f"Uploading volumes...")
            progress_cb = progress.update
        else:
            progress_cb = None

        item_names = []
        item_paths = []
        anns = []
        for item in self._items:
            if item.name in existing_names:
                new_name = generate_free_name(
                    existing_names, item.name, with_ext=True, extend_used_names=True
                )
                logger.warn(
                    f"Video with name '{item.name}' already exists, renaming to '{new_name}'"
                )
                item_names.append(new_name)
            else:
                item_names.append(item.name)
            item_paths.append(item.path)
            ann_path = f"{item.path}.json"
            i = 0
            while file_exists(ann_path):
                ann_path = f"{item.path}_{i}.json"
            ann = self.to_supervisely(item=item, meta=meta)
            dump_json_file(ann.to_json(), ann_path)
            anns.append(ann_path)

        vol_infos = api.volume.upload_nrrd_series_paths(
            dataset_id,
            item_names,
            item_paths,
            progress_cb=progress_cb,
        )
        for path in item_paths:
            silent_remove(path)
        vol_ids = [vol_info.id for vol_info in vol_infos]
        api.volume.annotation.upload_paths(vol_ids, anns, meta)
        for path in anns:
            silent_remove(path)

        if log_progress:
            progress.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")
