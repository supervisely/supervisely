import os
from typing import List

import supervisely.convert.image.sly.sly_image_helper as sly_image_helper
from supervisely import Annotation, Dataset, OpenMode, Project, ProjectMeta, logger
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import (
    JUNK_FILES,
    dirs_filter,
    file_exists,
    get_file_ext
)
from supervisely.io.json import load_json_file
from supervisely.project.project import find_project_dirs


class SLYImageConverter(ImageConverter):
    def __init__(self, input_data: str, labeling_interface: str) -> None:
        self._input_data: str = input_data
        self._items: List[ImageConverter.Item] = []
        self._meta: ProjectMeta = None
        self._labeling_interface = labeling_interface

    def __str__(self):
        return AvailableImageConverters.SLY

    def validate_labeling_interface(self) -> bool:
        return self._labeling_interface in ["default", "image_matting"]

    @property
    def ann_ext(self) -> str:
        return ".json"

    @property
    def key_file_ext(self) -> str:
        return ".json"

    def generate_meta_from_annotation(self, ann_path: str, meta: ProjectMeta) -> ProjectMeta:
        ann_json = load_json_file(ann_path)
        meta = sly_image_helper.get_meta_from_annotation(ann_json, meta)
        return meta

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta) -> bool:
        try:
            ann_json = load_json_file(ann_path)
            if "annotation" in ann_json:
                ann_json = ann_json["annotation"]
            ann = Annotation.from_json(ann_json, meta)
            return True
        except:
            return False

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            self._meta = ProjectMeta.from_json(load_json_file(key_file_path))
            return True
        except Exception:
            return False

    def validate_format(self) -> bool:
        if self.read_sly_project(self._input_data):
            return True

        if self.read_sly_dataset(self._input_data):
            return True

        detected_ann_cnt = 0
        images_list, ann_dict, img_meta_dict = [], {}, {}
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                dir_name = os.path.basename(root)
                if file == "meta.json":
                    is_valid = self.validate_key_file(full_path)
                    if is_valid:
                        continue

                ext = get_file_ext(full_path)
                if file in JUNK_FILES:  # add better check
                    continue
                elif ext in self.ann_ext:
                    if dir_name == "meta":
                        img_meta_dict[file] = full_path
                    else:
                        ann_dict[file] = full_path
                elif self.is_image(full_path):
                    images_list.append(full_path)

        if self._meta is not None:
            meta = self._meta
        else:
            meta = ProjectMeta()

        # create Items
        self._items = []
        for image_path in images_list:
            item = self.Item(image_path)
            ann_name = f"{item.name}.json"
            if ann_name in ann_dict:
                ann_path = ann_dict[ann_name]
                if self._meta is None:
                    meta = self.generate_meta_from_annotation(ann_path, meta)
                is_valid = self.validate_ann_file(ann_path, meta)
                if is_valid:
                    item.ann_data = ann_path
                    detected_ann_cnt += 1
            if ann_name in img_meta_dict:
                item.set_meta_data(img_meta_dict[ann_name])
            self._items.append(item)
        self._meta = meta
        return detected_ann_cnt > 0

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

        if item.ann_data is None:
            return item.create_empty_annotation()

        try:
            ann_json = load_json_file(item.ann_data)
            if "annotation" in ann_json:
                ann_json = ann_json["annotation"]
            if renamed_classes or renamed_tags:
                ann_json = sly_image_helper.rename_in_json(ann_json, renamed_classes, renamed_tags)
            return Annotation.from_json(ann_json, meta)
        except Exception as e:
            logger.warn(f"Failed to convert annotation: {repr(e)}")
            return item.create_empty_annotation()

    def read_sly_project(self, input_data: str) -> bool:
        try:
            self._items = []
            self._meta = None
            logger.debug("Trying to find Supervisely project format in the input data")
            project_dirs = [d for d in find_project_dirs(input_data)]
            if len(project_dirs) > 1:
                logger.info("Found multiple Supervisely projects")
            meta = ProjectMeta()
            for project_dir in project_dirs:
                project_fs = Project(project_dir, mode=OpenMode.READ)
                meta = meta.merge(project_fs.meta)
                for dataset in project_fs.datasets:
                    for name in dataset.get_items_names():
                        img_path, ann_path = dataset.get_item_paths(name)
                        meta_path = dataset.get_item_meta_path(name)
                        item = self.Item(img_path)
                        if file_exists(ann_path):
                            if self.validate_ann_file(ann_path, meta):
                                item.ann_data = ann_path
                        if file_exists(meta_path):
                            item.set_meta_data(meta_path)
                        self._items.append(item)
            if self.items_count > 0:
                self._meta = meta
                return True
            else:
                return False
        except Exception as e:
            logger.debug(f"Not a Supervisely project: {repr(e)}")
            return False

    def read_sly_dataset(self, input_data: str) -> bool:
        try:
            self._items = []
            self._meta = None
            logger.debug("Trying to read Supervisely datasets")

            def _check_function(path):
                try:
                    dataset_ds = Dataset(path, OpenMode.READ)
                    return len(dataset_ds.get_items_names()) > 0
                except:
                    return False

            meta = ProjectMeta()
            dataset_dirs = [d for d in dirs_filter(input_data, _check_function)]
            for dataset_dir in dataset_dirs:
                dataset_ds = Dataset(dataset_dir, OpenMode.READ)
                for name in dataset_ds.get_items_names():
                    img_path, ann_path = dataset_ds.get_item_paths(name)
                    meta_path = dataset_ds.get_item_meta_path(name)
                    item = self.Item(img_path)
                    if file_exists(ann_path):
                        meta = self.generate_meta_from_annotation(ann_path, meta)
                        if self.validate_ann_file(ann_path, meta):
                            item.ann_data = ann_path
                    if file_exists(meta_path):
                        item.set_meta_data(meta_path)
                    self._items.append(item)

            if self.items_count > 0:
                self._meta = meta
                return True
            else:
                return False
        except Exception as e:
            logger.debug(f"Failed to read Supervisely datasets: {repr(e)}")
            return False
