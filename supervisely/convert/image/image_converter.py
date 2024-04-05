from typing import List, Optional, Tuple, Union

from tqdm import tqdm

import supervisely.imaging.image as image
from supervisely import (
    Annotation,
    Api,
    ProjectMeta,
    batched,
    generate_free_name,
    logger,
)
from supervisely.convert.base_converter import BaseConverter
from supervisely.imaging.image import SUPPORTED_IMG_EXTS
from supervisely.io.json import load_json_file


class ImageConverter(BaseConverter):
    allowed_exts = SUPPORTED_IMG_EXTS

    class Item(BaseConverter.BaseItem):
        def __init__(
            self,
            item_path: str,
            ann_data: Union[str, dict] = None,
            meta_data: Union[str, dict] = None,
            shape: Union[Tuple, List] = None,
            custom_data: Optional[dict] = None,
        ):
            self._path: str = item_path
            self._ann_data: Union[str,] = ann_data
            self._meta_data: Union[str, dict] = meta_data
            self._type: str = "image"
            # TODO: Fix the issue to open different images (tiff, multichannel, nrrd) and remove the try-except block.
            try:
                if shape is None:
                    img = image.read(item_path)
                    self._shape: Union[Tuple, List] = img.shape[:2]
                else:
                    self._shape: Union[Tuple, List] = shape
            except Exception as e:
                logger.warning(f"Failed to read image shape: {e}, shape is set to [0, 0]")
                self._shape = [0, 0]
            # TODO: End of the block with the issue.
            self._custom_data: dict = custom_data if custom_data is not None else {}

        @property
        def meta(self) -> Union[str, dict]:
            return self._meta_data

        def set_meta_data(self, meta_data: Union[str, dict]) -> None:
            self._meta_data = meta_data

        def create_empty_annotation(self) -> Annotation:
            return Annotation(self._shape)

    def __init__(
        self,
        input_data: str,
        labeling_interface: str,
    ):
        self._input_data: str = input_data
        self._meta: ProjectMeta = None
        self._items: List[self.Item] = []
        self._labeling_interface: str = labeling_interface
        self._converter = self._detect_format()

    @property
    def format(self) -> str:
        return self._converter.format

    @property
    def ann_ext(self) -> str:
        return None

    @property
    def key_file_ext(self) -> str:
        return None

    def get_meta(self) -> ProjectMeta:
        return self._meta

    def get_items(self) -> List[BaseConverter.BaseItem]:
        return super().get_items()

    @staticmethod
    def validate_ann_file(ann_path: str, meta: ProjectMeta = None) -> bool:
        return False

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
    ) -> None:
        """Upload converted data to Supervisely"""

        dataset = api.dataset.get_info_by_id(dataset_id)
        existing_names = set([img.name for img in api.image.get_list(dataset.id)])
        if self._meta is not None:
            curr_meta = self._meta
        else:
            curr_meta = ProjectMeta()
        meta_json = api.project.get_meta(dataset.project_id)
        meta = ProjectMeta.from_json(meta_json)
        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(meta, curr_meta)

        api.project.update_meta(dataset.project_id, meta)

        if log_progress:
            progress = tqdm(total=self.items_count, desc="Uploading images...")
            progress_cb = progress.update
        else:
            progress_cb = None

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            item_metas = []
            anns = []
            for item in batch:
                ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)

                if item.name in existing_names:
                    new_name = generate_free_name(
                        existing_names, item.name, with_ext=True, extend_used_names=True
                    )
                    logger.warn(
                        f"Image with name '{item.name}' already exists, renaming to '{new_name}'"
                    )
                    item_names.append(new_name)
                else:
                    item_names.append(item.name)
                item_paths.append(item.path)
                item_metas.append(load_json_file(item.meta) if item.meta else {})
                anns.append(ann)

            img_infos = api.image.upload_paths(
                dataset_id, item_names, item_paths, progress_cb, item_metas
            )
            img_ids = [img_info.id for img_info in img_infos]
            api.annotation.upload_anns(img_ids, anns)

        if log_progress:
            progress.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")


# @TODO:
# COCO
# [x] - Implement Skeleton support
# [x] - Implement binding of bboxes and masks
# [ ] - Implement detailed coco label validation
# Supervisely
# [x] - Implement keypoints generation (when meta not found)
# [ ] - Add ann keys validation to method `generate_meta_from_annotation()`
# [ ] - Add ann keys validation to method `generate_meta_from_annotation()`
