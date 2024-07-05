import os

from pathlib import Path
from typing import Optional, OrderedDict, Union

from supervisely import (
    Api,
    batched,
    generate_free_name,
    is_development,
    logger,
)
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely.convert.base_converter import BaseConverter
from supervisely.io.fs import get_file_ext, get_file_name_with_ext
from supervisely.volume.volume import ALLOWED_VOLUME_EXTENSIONS, read_nrrd_serie_volume


class VolumeConverter(BaseConverter):
    allowed_exts = ALLOWED_VOLUME_EXTENSIONS
    modality = "volumes"

    class Item(BaseConverter.BaseItem):
        def __init__(
            self,
            item_path: str,
            ann_data: str = None,
            shape: tuple = None,
            custom_data: Optional[dict] = None,
            volume_meta: Union[dict, OrderedDict] = None,
            mask_dir: str = None,
            interpolation_dir: str = None,
        ):
            self._path: str = item_path
            self._name: str = None
            self._ann_data: str = ann_data
            if volume_meta is None:
                sitk_volume, meta = read_nrrd_serie_volume(item_path)
                self._volume_meta = meta
            else:
                self._volume_meta: Union[dict, OrderedDict] = volume_meta
            if shape is None:
                self._shape: tuple = (
                    self._volume_meta["dimensionsIJK"]["y"],
                    self._volume_meta["dimensionsIJK"]["x"],
                    self._volume_meta["dimensionsIJK"]["z"],
                )
            else:
                self._shape: tuple = shape

            self._mask_dir: str = mask_dir
            self._interpolation_dir: str = interpolation_dir
            self._type: str = "volume"
            self._custom_data: dict = custom_data if custom_data is not None else {}

        @property
        def volume_meta(self) -> Union[dict, OrderedDict]:
            return self._volume_meta

        @property
        def mask_dir(self) -> str:
            return self._mask_dir

        @property
        def interpolation_dir(self) -> str:
            return self._interpolation_dir

        @volume_meta.setter
        def volume_meta(self, meta: Union[dict, OrderedDict]) -> None:
            self._volume_meta = meta

        @mask_dir.setter
        def mask_dir(self, mask_dir: str) -> None:
            self._mask_dir = mask_dir

        @interpolation_dir.setter
        def interpolation_dir(self, interpolation_dir: str) -> None:
            self._interpolation_dir = interpolation_dir

        def create_empty_annotation(self) -> VolumeAnnotation:
            return VolumeAnnotation(self._volume_meta)

    @property
    def format(self):
        return self._converter.format

    @property
    def ann_ext(self):
        return None

    @property
    def key_file_ext(self):
        return None

    @staticmethod
    def validate_ann_file(ann_path, meta=None):
        return False

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 1,
        log_progress=True,
    ):
        """Upload converted data to Supervisely"""

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        existing_names = set([vol.name for vol in api.volume.get_list(dataset_id)])

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading volumes...")
        else:
            progress_cb = None

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            anns = []
            mask_dirs = []
            interpolation_dirs = []
            for item in batch:
                ext = get_file_ext(item.path)
                if ext.lower() != ext:
                    new_volume_path = Path(item.path).with_suffix(ext.lower()).as_posix()
                    os.rename(item.path, new_volume_path)
                    item.path = new_volume_path
                item.name = get_file_name_with_ext(item.path)
                item.name = generate_free_name(
                    existing_names, item.name, with_ext=True, extend_used_names=True
                )
                item_names.append(item.name)
                item_paths.append(item.path)
                anns.append(item.ann_data)
                mask_dirs.append(item.mask_dir)
                interpolation_dirs.append(item.interpolation_dir)

                ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
                anns.append(ann)

            vol_infos = api.volume.upload_nrrd_series_paths(
                dataset_id,
                item_names,
                item_paths,
            )
            vol_ids = [vol_info.id for vol_info in vol_infos]
            if all(ann is not None for ann in anns):
                api.volume.annotation.upload_paths(
                    vol_ids, anns, meta, interpolation_dirs=interpolation_dirs, mask_dirs=mask_dirs
                )

            if log_progress:
                progress_cb(len(batch))

        if log_progress:
            if is_development():
                progress.close()
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")
