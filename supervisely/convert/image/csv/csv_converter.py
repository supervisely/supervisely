import os
from typing import List

import supervisely.convert.image.csv.csv_helper as csv_helper
from supervisely import Annotation, ProjectMeta, TagCollection, logger
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import get_file_ext


class CSVConverter(ImageConverter):
    def __init__(self, input_data: str):
        self._input_data: str = input_data
        self._items: List[ImageConverter.Item] = []
        self._meta: ProjectMeta = None
        self._csv_table = None

    def __str__(self):
        return AvailableImageConverters.CSV

    @property
    def key_file_ext(self) -> str:
        return [".csv", ".txt"]

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            self._csv_table, self._meta, csv_reader = csv_helper.validate_and_collect_items(
                key_file_path
            )
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
            self._items = self.collect_items(self._csv_table)
            return True
        else:
            return False

    def collect_items(self):
        items = []
        for row in self._csv_table:
            item_path = row[0]
            ann_data = row[1]
            item = ImageConverter.Item(
                item_path=item_path,
                ann_data=ann_data,
            )
            items.append(item)
        return items

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
