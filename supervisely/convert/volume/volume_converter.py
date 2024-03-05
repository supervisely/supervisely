from typing import List, Optional, OrderedDict, Union

import nrrd
from tqdm import tqdm

from supervisely import Api, batched, generate_free_name, logger, ProjectMeta, VolumeAnnotation
from supervisely.api.module_api import ApiField
from supervisely.convert.base_converter import BaseConverter
from supervisely.io.json import load_json_file
from supervisely.volume.volume import read_nrrd_serie_volume


class VolumeConverter(BaseConverter):
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

        def create_empty_annotation(self) -> VolumeAnnotation:
            return VolumeAnnotation(self._volume_meta)

        def set_volume_meta(self, meta: dict) -> None:
            self._volume_meta = meta

        def set_mask_dir(self, mask_dir: str) -> None:
            self._mask_dir = mask_dir

        def set_interpolation_dir(self, interpolation_dir: str) -> None:
            self._interpolation_dir = interpolation_dir

    def __init__(self, input_data: str):
        self._input_data: str = input_data
        self._meta: ProjectMeta = None
        self._items: List[self.Item] = []
        self._converter = self._detect_format()
        self._batch_size: int = 1

    @property
    def format(self):
        return self.converter.format

    @property
    def ann_ext(self):
        return None

    @property
    def key_file_ext(self):
        return None

    def get_meta(self) -> ProjectMeta:
        return self._meta

    def get_items(self) -> List[BaseConverter.BaseItem]:
        return self._items

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

        if self.items_count == 0:
            raise RuntimeError("Nothing to upload. Check the input data.")

        dataset = api.dataset.get_info_by_id(dataset_id)
        existing_names = set([vol.name for vol in api.image.get_list(dataset.id)])
        meta_json = api.project.get_meta(dataset.project_id)
        meta = ProjectMeta.from_json(meta_json)
        meta = meta.merge(self._meta)

        api.project.update_meta(dataset.project_id, meta)

        if log_progress:
            progress = tqdm(total=self.items_count, desc=f"Uploading volumes...")
            progress_cb = progress.update
        else:
            progress_cb = None

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            anns = []
            mask_dirs = []
            interpolation_dirs = []
            for item in batch:
                if item.name in existing_names:
                    new_name = generate_free_name(
                        existing_names, item.name, with_ext=True, extend_used_names=True
                    )
                    logger.warn(
                        f"Item with name '{item.name}' already exists, renaming to '{new_name}'"
                    )
                    item_names.append(new_name)
                else:
                    item_names.append(item.name)
                item_paths.append(item.path)
                anns.append(item.ann_data)
                mask_dirs.append(item.mask_dir)
                interpolation_dirs.append(item.interpolation_dir)

                # ann = self.to_supervisely(item, meta)
                # anns.append(ann)

            # for .dcm
            # api.volume.upload_dicom_serie_paths()

            vol_infos = api.volume.upload_nrrd_series_paths(
                dataset_id,
                item_names,
                item_paths,
                progress_cb=progress_cb,
            )
            vol_ids = [vol_info.id for vol_info in vol_infos]
            api.volume.annotation.upload_paths(
                vol_ids, anns, meta, interpolation_dirs=interpolation_dirs, mask_dirs=mask_dirs
            )

        if log_progress:
            progress.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")


# @TODO:
# [ ] - add DICOM support
# [ ] - add support volumes without .ext
