import os
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
from supervisely.io.fs import download
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
            local: bool = True,
            tf: bool = False,
        ):
            self._path: str = item_path
            self._ann_data: Union[str,] = ann_data
            self._meta_data: Union[str, dict] = meta_data
            self._type: str = "image"
            self._local: bool = local
            self._tf: bool = tf
            if shape is None and local is True:
                img = image.read(item_path)
                self._shape: Union[Tuple, List] = img.shape[:2]
            else:
                self._shape: Union[Tuple, List] = shape
            self._custom_data: dict = custom_data if custom_data is not None else {}

        @property
        def meta(self) -> Union[str, dict]:
            return self._meta_data

        @property
        def local(self) -> bool:
            return self._local

        @property
        def tf(self) -> bool:
            return self._tf

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
            progress = tqdm(total=self.items_count, desc=f"Uploading images...")
            progress_cb = progress.update
        else:
            progress_cb = None

        local_items = [item for item in self._items if item.local]
        non_local_items = [item for item in self._items if not item.local]
        items_dict = {"local": local_items, "non_local": non_local_items}

        for key in items_dict:
            for batch in batched(items_dict[key], batch_size=batch_size):
                item_names = []
                item_paths = []
                item_metas = []
                anns = []
                if key == "non_local":
                    is_tf = []
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
                    if key == "non_local":
                        is_tf.append(item.tf)

                if key == "local":
                    img_infos = api.image.upload_paths(
                        dataset_id, item_names, item_paths, progress_cb, item_metas
                    )
                if key == "non_local":
                    img_infos = self.upload_remote_images(
                        api, dataset_id, item_names, item_paths, is_tf, progress_cb
                    )
                img_ids = [img_info.id for img_info in img_infos]
                api.annotation.upload_anns(img_ids, anns)

        if log_progress:
            progress.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")

    def process_remote_image(
        self,
        api: Api,
        team_id: int,
        image_path: str,
        image_name: str,
        dataset_id: int,
        is_tf,
        progress_cb: Optional[callable] = None,
        force_metadata: bool = True,
    ):
        image_path = image_path.strip()
        if is_tf:
            if not api.file.exists(team_id, image_path.lstrip("/")):
                logger.warn(f"File {image_path} not found in Team Files. Skipping...")
            tf_image_info = api.file.list(team_id, image_path.lstrip("/"))
            image_path = tf_image_info[0]["fullStorageUrl"]

        extension = os.path.splitext(image_path)[1]
        if not extension:
            logger.warn(f"Image [{image_path}] doesn't have extension in link")

        image_info = api.image.upload_link(
            dataset_id=dataset_id,
            name=image_name,
            link=image_path,
            force_metadata_for_links=force_metadata,
        )
        if progress_cb is not None:
            progress_cb(1)
        return image_info

    def upload_remote_images(
        self,
        api: Api,
        dataset_id: int,
        image_names: list,
        image_paths: list,
        is_tf: list,
        progress_cb: Optional[callable] = None,
        force_metadata: bool = True,
    ):
        image_infos = [
            self.process_remote_image(
                api,
                self.team_id,
                image_path,
                image_name,
                dataset_id,
                tf,
                progress_cb,
                force_metadata,
            )
            for image_name, image_path, tf in zip(image_names, image_paths, is_tf)
        ]
        return image_infos


# @TODO:
# COCO
# [x] - Implement Skeleton support
# [x] - Implement binding of bboxes and masks
# [ ] - Implement detailed coco label validation
# Supervisely
# [x] - Implement keypoints generation (when meta not found)
# [ ] - Add ann keys validation to method `generate_meta_from_annotation()`
