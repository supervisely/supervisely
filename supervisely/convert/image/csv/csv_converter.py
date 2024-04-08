from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

from tqdm import tqdm

import supervisely.convert.image.csv.csv_helper as csv_helper
from supervisely import (
    Annotation,
    Api,
    ProjectMeta,
    TagCollection,
    batched,
    generate_free_name,
    logger,
)
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import get_file_ext
from supervisely.io.json import load_json_file


class CSVConverter(ImageConverter):

    class Item(ImageConverter.Item):

        def __init__(
            self,
            item_path: str,
            ann_data: Optional[str] = None,
            meta_data: str | dict = None,
            shape: Tuple | List = None,
            custom_data: dict | None = None,
            team_files: bool = False,
        ):
            self._path: str = item_path
            self._ann_data: Union[str,] = ann_data
            self._meta_data: Union[str, dict] = meta_data
            self._type: str = "image"
            if shape is None:
                self._shape: Union[Tuple, List] = [None, None]
            else:
                self._shape: Union[Tuple, List] = shape
            self._custom_data: dict = custom_data if custom_data is not None else {}
            self._team_files = team_files

        @property
        def team_files(self) -> bool:
            return self._team_files

        @property
        def shape(self) -> bool:
            return self._shape

        @shape.setter
        def shape(self, value: Tuple | List):
            self._shape = value

    def __init__(self, input_data: str):
        self._input_data: str = input_data
        self._items: List[ImageConverter.Item] = []
        self._meta: ProjectMeta = None
        self._csv_reader = None
        self._team_id = None
        # self._labeling_interface: str = labeling_interface

    def __str__(self):
        return AvailableImageConverters.CSV

    @property
    def key_file_ext(self) -> str:
        return [".csv", ".txt"]

    @property
    def team_id(self) -> int:
        return self._team_id

    @team_id.setter
    def team_id(self, value: int):
        if not isinstance(value, int):
            raise TypeError("team_id must be an integer")
        self._team_id = value

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            self._meta, self._csv_reader = csv_helper.validate_and_collect_items(key_file_path)
            return True
        except Exception:
            return False

    def validate_format(self) -> bool:
        files = [
            f
            for f in os.listdir(self._input_data)
            if os.path.isfile(os.path.join(self._input_data, f))
        ]
        valid_files = [f for f in files if os.path.splitext(f)[1] in self.key_file_ext]

        if len(valid_files) != 1:
            return False

        full_path = os.path.join(self._input_data, valid_files[0])
        if get_file_ext(full_path) == ".txt":
            csv_full_path = os.path.splitext(full_path)[0] + ".csv"
            csv_helper.convert_txt_to_csv(full_path, csv_full_path)
            full_path = csv_full_path

        if self.validate_key_file(full_path):
            self.collect_items()
            return True
        else:
            return False

    def collect_items(self):
        for row in self._csv_reader:
            possible_paths = (
                csv_helper.possible_image_path_col_names + csv_helper.possible_image_url_col_names
            )
            for possible_path in possible_paths:
                if possible_path in row:
                    item_path = row.get(possible_path)
                    if possible_path in csv_helper.possible_image_path_col_names:
                        team_files = True
                    else:
                        team_files = False
                    break
            if item_path is None:
                logger.warn(f"Failed to find image path in row: {row}. Skipping.")
                continue
            ann_data = row.get("tag")
            item = CSVConverter.Item(
                item_path=item_path,
                ann_data=ann_data,
                team_files=team_files,
            )
            self._items.append(item)

    def to_supervisely(
        self,
        item: CSVConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> Annotation:
        """Convert to Supervisely format."""

        if meta is None:
            meta = self._meta

        try:
            if item.ann_data is None:
                return item.create_empty_annotation()
            tag_names = item.ann_data.split(csv_helper.TAGS_DELIMITER)
            tag_metas = [
                meta.get_tag_meta(tag_name.strip())
                for tag_name in tag_names
                if tag_name.strip() != ""
            ]

            tag_collection = TagCollection(tag_metas)

            if item.shape:
                ann = Annotation(item.shape).add_tags(tag_collection)
            else:
                ann = Annotation(img_size=[None, None]).add_tags(tag_collection)
            ann_json = ann.to_json()
            if "annotation" in ann_json:
                ann_json = ann_json["annotation"]
            if renamed_classes or renamed_tags:
                ann_json = csv_helper.rename_in_json(ann_json, renamed_classes, renamed_tags)
            return Annotation.from_json(ann_json, meta)
        except Exception as e:
            logger.warn(f"Failed to convert annotation: {repr(e)}")
            return item.create_empty_annotation()

    def process_remote_image(
        self,
        api: Api,
        team_id: int,
        image_path: str,
        image_name: str,
        dataset_id: int,
        is_team_file: bool,
        progress_cb: Optional[callable] = None,
        force_metadata: bool = True,
    ):
        image_path = image_path.strip()
        if is_team_file:
            if not api.file.exists(team_id, image_path):
                logger.warn(f"File {image_path} not found in Team Files. Skipping...")
                return None
            team_file_image_info = api.file.list(team_id, image_path)
            image_path = team_file_image_info[0]["fullStorageUrl"]
            if not image_path:
                logger.warn(f"Failed to get full storage URL for file '{image_path}'. Skipping...")
                return None

        extension = os.path.splitext(image_path)[1]
        if not extension:
            logger.warn(f"Image [{image_path}] doesn't have extension in path\url")

        try:
            image_info = api.image.upload_link(
                dataset_id=dataset_id,
                name=image_name,
                link=image_path,
                force_metadata_for_links=force_metadata,
            )
        except Exception:
            logger.warn(f"Failed to upload image {image_name}. Skipping...")
            return None
        if progress_cb is not None:
            progress_cb(1)
        return image_info

    def upload_remote_images(
        self,
        api: Api,
        dataset_id: int,
        image_names: List[str],
        image_paths: List[str],
        is_team_files: List[bool],
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
                is_team_file,
                progress_cb,
                force_metadata,
            )
            for image_name, image_path, is_team_file in zip(image_names, image_paths, is_team_files)
        ]
        return image_infos

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
            progress = tqdm(total=self.items_count, desc=f"Uploading images")
            progress_cb = progress.update
        else:
            progress_cb = None

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            item_metas = []
            is_team_files = []
            anns = []

            for item in batch:
                item: CSVConverter.Item

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
                is_team_files.append(item.team_files)

            img_infos = self.upload_remote_images(
                api, dataset_id, item_names, item_paths, is_team_files, progress_cb
            )
            img_ids = [img_info.id for img_info in img_infos if img_info is not None]

            for item, info in zip(batch, img_infos):
                if info is None:
                    continue
                if item.name != info.name:
                    logger.warn(
                        f"Batched image name '{item.name}' doesn't match uploaded image name '{info.name}'"
                    )
                item: CSVConverter.Item
                if item.shape is None or item.shape == [None, None]:
                    item.set_shape([info.height, info.width])
                ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
                anns.append(ann)

            if log_progress:
                progress_ann = tqdm(total=len(anns), desc=f"Uploading annotations")
                progress_ann_cb = progress_ann.update
            else:
                progress_ann_cb = None
            api.annotation.upload_anns(img_ids, anns, progress_ann_cb)

        if log_progress:
            progress.close()
            progress_ann.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")
