import os
from typing import List

import supervisely.convert.image.csv.csv_helper as csv_helper
from supervisely import Annotation, ProjectMeta, TagCollection, logger
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import get_file_ext


class CSVConverter(ImageConverter):

    def __init__(self, input_data: str, labeling_interface: str):
        self._input_data: str = input_data
        self._items: List[ImageConverter.Item] = []
        self._meta: ProjectMeta = None
        self._csv_reader = None
        self._team_id = None
        self._labeling_interface: str = labeling_interface

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
                        tf = True
                    else:
                        tf = False
                    break
            if item_path is None:
                logger.warn(f"Failed to find image path in row: {row}. Skipping.")
                continue
            ann_data = row.get("tag")
            item = ImageConverter.Item(
                item_path=item_path,
                ann_data=ann_data,
                local=False,
                tf=tf,
            )
            self._items.append(item)

    def to_supervisely(
        self,
        item: ImageConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> Annotation:
        """Convert to Supervisely format."""

        if meta is None:
            meta = self._meta

        try:
            tag_names = item.ann_data.split(csv_helper.TAGS_DELIMITER)
            tag_metas = [
                meta.get_tag_meta(tag_name.strip())
                for tag_name in tag_names
                if tag_name.strip() != ""
            ]

            tag_collection = TagCollection(tag_metas)

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
