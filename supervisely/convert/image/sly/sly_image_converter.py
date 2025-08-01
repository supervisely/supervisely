import os
from typing import Dict, Optional

import supervisely.convert.image.sly.sly_image_helper as sly_image_helper
from supervisely import (
    Annotation,
    Dataset,
    Label,
    OpenMode,
    Project,
    ProjectMeta,
    Rectangle,
    logger,
)
from supervisely._utils import generate_free_name, is_development
from supervisely.api.api import Api
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.image.image_helper import validate_image_bounds
from supervisely.io.fs import dirs_filter, file_exists, get_file_ext, get_file_name
from supervisely.io.json import load_json_file
from supervisely.project.project import find_project_dirs
from supervisely.project.project import upload_project as upload_project_fs
from supervisely.project.project_settings import LabelingInterface

DATASET_ITEMS = "items"
NESTED_DATASETS = "datasets"


class SLYImageConverter(ImageConverter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._project_structure = None
        self._supports_links = True
        self._blob_project = False

    def __str__(self):
        return AvailableImageConverters.SLY

    @property
    def blob_project(self) -> bool:
        return self._blob_project

    @blob_project.setter
    def blob_project(self, value: bool):
        self._blob_project = value

    @property
    def ann_ext(self) -> str:
        return ".json"

    @property
    def key_file_ext(self) -> str:
        return ".json"

    def validate_labeling_interface(self) -> bool:
        return self._labeling_interface in [
            LabelingInterface.DEFAULT,
            LabelingInterface.IMAGE_MATTING,
            LabelingInterface.FISHEYE,
            LabelingInterface.MULTIVIEW,
        ]

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
        if self.upload_as_links and self._supports_links:
            self._download_remote_ann_files()
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
                if ext in self.ann_ext:
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
            json_name, json_name_noext = f"{item.name}.json", f"{get_file_name(item.name)}.json"
            ann_path = ann_dict.get(json_name, ann_dict.get(json_name_noext))
            if ann_path:
                if self._meta is None:
                    meta = self.generate_meta_from_annotation(ann_path, meta)
                is_valid = self.validate_ann_file(ann_path, meta)
                if is_valid:
                    item.ann_data = ann_path
            meta_path = img_meta_dict.get(json_name, img_meta_dict.get(json_name_noext))
            if meta_path:
                item.set_meta_data(meta_path)
            if item.ann_data is not None or item.meta is not None:
                detected_ann_cnt += 1
            self._items.append(item)
        self._meta = meta
        return detected_ann_cnt > 0

    def to_supervisely(
        self,
        item: ImageConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: Optional[Dict[str, str]] = None,
        renamed_tags: Optional[Dict[str, str]] = None,
    ) -> Annotation:
        """Convert to Supervisely format."""
        if meta is None:
            meta = self._meta

        if item.ann_data is None:
            if self._upload_as_links:
                item.set_shape([None, None])
            return item.create_empty_annotation()

        try:
            ann_json = load_json_file(item.ann_data)
            if "annotation" in ann_json:
                ann_json = ann_json["annotation"]
            if renamed_classes or renamed_tags:
                ann_json = sly_image_helper.rename_in_json(ann_json, renamed_classes, renamed_tags)
            img_size = ann_json["size"]  # dict { "width": 1280, "height": 720 }
            if "width" not in img_size or "height" not in img_size:
                raise ValueError("Invalid image size in annotation JSON")
            img_size = (img_size["height"], img_size["width"])
            labels = validate_image_bounds(
                [Label.from_json(obj, meta) for obj in ann_json["objects"]],
                Rectangle.from_size(img_size),
            )
            return Annotation.from_json(ann_json, meta).clone(labels=labels)
        except Exception as e:
            logger.warning(f"Failed to convert annotation: {repr(e)}")
            return item.create_empty_annotation()

    def read_sly_project(self, input_data: str) -> bool:
        try:
            self._items = []
            project = {}
            ds_cnt = 0
            self._meta = None
            logger.debug("Trying to find Supervisely project format in the input data")
            project_dirs = [d for d in find_project_dirs(input_data)]
            if len(project_dirs) > 1:
                logger.info("Found multiple possible Supervisely projects in the input data")
            elif len(project_dirs) == 1:
                logger.info("Possible Supervisely project found in the input data")
            meta = None
            for project_dir in project_dirs:
                project_fs = Project(project_dir, mode=OpenMode.READ)
                if len(project_fs.blob_files) > 0:
                    self.blob_project = True
                    logger.info("Found blob files in the project, skipping")
                    continue

                if meta is None:
                    meta = project_fs.meta
                else:
                    meta = meta.merge(project_fs.meta)
                for dataset in project_fs.datasets:
                    ds_items = []
                    for name in dataset.get_items_names():
                        img_path, ann_path = dataset.get_item_paths(name)
                        meta_path = dataset.get_item_meta_path(name)
                        item = self.Item(img_path)
                        if file_exists(ann_path):
                            if self.validate_ann_file(ann_path, meta):
                                item.ann_data = ann_path
                        if file_exists(meta_path):
                            item.set_meta_data(meta_path)
                        ds_items.append(item)
                    if len(ds_items) > 0:
                        parts = dataset.name.split("/")
                        curr_ds = project.setdefault(
                            parts[0], {DATASET_ITEMS: [], NESTED_DATASETS: {}}
                        )
                        for part in parts[1:]:
                            curr_ds = curr_ds[NESTED_DATASETS].setdefault(
                                part, {DATASET_ITEMS: [], NESTED_DATASETS: {}}
                            )
                        curr_ds[DATASET_ITEMS].extend(ds_items)
                        ds_cnt += 1
                        self._items.extend(ds_items)
            if self.items_count > 0:
                self._meta = meta
                if ds_cnt > 1:  # multiple datasets
                    self._project_structure = project
                return True
            elif self.blob_project:
                return True
            else:
                return False
        except Exception as e:
            logger.debug(f"Not a Supervisely project: {repr(e)}")
            return False

    def read_sly_dataset(self, input_data: str) -> bool:
        try:
            self._items = []
            project = {}
            ds_cnt = 0
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
                dataset_fs = Dataset(dataset_dir, OpenMode.READ)
                ds_items = []
                for name in dataset_fs.get_items_names():
                    img_path, ann_path = dataset_fs.get_item_paths(name)
                    meta_path = dataset_fs.get_item_meta_path(name)
                    item = self.Item(img_path)
                    if file_exists(ann_path):
                        meta = self.generate_meta_from_annotation(ann_path, meta)
                        if self.validate_ann_file(ann_path, meta):
                            item.ann_data = ann_path
                    if file_exists(meta_path):
                        item.set_meta_data(meta_path)
                    ds_items.append(item)
                if len(ds_items) > 0:
                    parts = dataset_fs.name.split("/")
                    curr_ds = project.setdefault(parts[0], {DATASET_ITEMS: [], NESTED_DATASETS: {}})
                    for part in parts[1:]:
                        curr_ds = curr_ds[NESTED_DATASETS].setdefault(
                            part, {DATASET_ITEMS: [], NESTED_DATASETS: {}}
                        )
                    curr_ds[DATASET_ITEMS].extend(ds_items)
                    ds_cnt += 1
                    self._items.extend(ds_items)

            if self.items_count > 0:
                self._meta = meta
                if ds_cnt > 1:  # multiple datasets
                    self._project_structure = project
                return True
            else:
                return False
        except Exception as e:
            logger.debug(f"Failed to read Supervisely datasets: {repr(e)}")
            return False

    def upload_dataset(
        self, api: Api, dataset_id: int, batch_size: int = 50, log_progress=True
    ) -> None:

        if self._project_structure:
            self.upload_project(api, dataset_id, batch_size, log_progress)
        elif self.blob_project:
            dataset_info = api.dataset.get_info_by_id(dataset_id, raise_error=True)
            project_dirs = [d for d in find_project_dirs(self._input_data)]
            if len(project_dirs) == 0:
                raise RuntimeError(
                    "Failed to find Supervisely project with blobs in the input data"
                )
            project_dir = project_dirs[0]
            if len(project_dirs) > 1:
                logger.info(
                    "Found multiple possible Supervisely projects with blobs in the input data. "
                    f"Only the first one will be uploaded: {project_dir}"
                )
            upload_project_fs(
                dir=project_dir,
                api=api,
                workspace_id=dataset_info.workspace_id,
                log_progress=log_progress,
                project_id=dataset_info.project_id,
            )
        else:
            super().upload_dataset(api, dataset_id, batch_size, log_progress)

    def upload_project(
        self, api: Api, dataset_id: int, batch_size: int = 50, log_progress=True
    ) -> None:
        dataset_info = api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_id = dataset_info.project_id
        existing_datasets = api.dataset.get_list(project_id, recursive=True)
        existing_datasets = {ds.name for ds in existing_datasets}

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading project")
        else:
            progress, progress_cb = None, None

        logger.info("Uploading project structure")

        def _upload_project(
            project_structure: Dict,
            project_id: int,
            dataset_id: int,
            parent_id: Optional[int] = None,
            first_dataset=False,
        ):
            for ds_name, value in project_structure.items():
                ds_name = generate_free_name(existing_datasets, ds_name, extend_used_names=True)
                if first_dataset:
                    first_dataset = False
                    api.dataset.update(dataset_id, ds_name)  # rename first dataset
                else:
                    dataset_id = api.dataset.create(project_id, ds_name, parent_id=parent_id).id

                items = value.get(DATASET_ITEMS, [])
                nested_datasets = value.get(NESTED_DATASETS, {})
                logger.info(
                    f"Dataset: {ds_name}, items: {len(items)}, nested datasets: {len(nested_datasets)}"
                )
                if items:
                    super(SLYImageConverter, self).upload_dataset(
                        api, dataset_id, batch_size, entities=items, progress_cb=progress_cb
                    )

                if nested_datasets:
                    _upload_project(nested_datasets, project_id, dataset_id, dataset_id)

        _upload_project(self._project_structure, project_id, dataset_id, first_dataset=True)

        if is_development() and progress is not None:
            progress.close()
