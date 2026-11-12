import os
from typing import Dict, List, Optional

import supervisely.convert.pointcloud.sly.sly_pointcloud_helper as helpers
from supervisely import PointcloudAnnotation, ProjectMeta, logger
from supervisely._utils import generate_free_name, is_development
from supervisely.api.api import Api
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.imaging.image import is_valid_ext
from supervisely.io.fs import get_file_ext, get_file_name, get_file_name_with_ext
from supervisely.io.json import load_json_file
from supervisely.pointcloud.pointcloud import validate_ext as validate_pcd_ext

DATASET_ITEMS = "items"
NESTED_DATASETS = "datasets"


class SLYPointcloudConverter(PointcloudConverter):
    """Imports Supervisely point cloud project format (PCD + JSON annotations) from local folder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._supports_links = True
        # Set when the input holds more than one dataset in the project layout:
        #   { ds_name: { "items": [Item, ...], "datasets": { child_name: {...} } } }
        self._project_structure = None

    def __str__(self) -> str:
        return AvailablePointcloudConverters.SLY

    @property
    def ann_ext(self) -> str:
        return ".json"

    @property
    def key_file_ext(self) -> str:
        return ".json"

    def generate_meta_from_annotation(self, ann_path: str, meta: ProjectMeta) -> ProjectMeta:
        meta = helpers.get_meta_from_annotation(ann_path, meta)
        return meta

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta) -> bool:
        try:
            ann_json = load_json_file(ann_path)
            if "annotation" in ann_json:
                ann_json = ann_json["annotation"]
            ann = PointcloudAnnotation.from_json(ann_json, meta)
            return True
        except Exception as e:
            return False

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            self._meta = ProjectMeta.from_json(load_json_file(key_file_path))
            return True
        except Exception:
            return False

    def validate_format(self) -> bool:
        if self.upload_as_links and self.supports_links:
            self._download_remote_ann_files()
        pcd_list, ann_dict, rimg_dict, rimg_ann_dict = [], {}, {}, {}
        for root, _, files in os.walk(self._input_data):
            dir_name = os.path.basename(root)
            for file in files:
                full_path = os.path.join(root, file)
                if file == "key_id_map.json":
                    continue
                if file == "meta.json":
                    is_valid = self.validate_key_file(full_path)
                    if is_valid:
                        continue

                ext = get_file_ext(full_path)
                if ext in self.ann_ext:
                    parent_dir_name = os.path.basename(os.path.dirname(root))
                    possible_dirs = ["images", "related images", "photo context"]
                    if any(
                        p.replace("_", " ") in possible_dirs for p in [dir_name, parent_dir_name]
                    ):
                        rimg_ann_dict[file] = full_path
                    else:
                        ann_dict[file] = full_path
                elif self._is_image_file(full_path) or (self.upload_as_links and is_valid_ext(ext)):
                    if root not in rimg_dict:
                        rimg_dict[root] = []
                    rimg_dict[root].append(full_path)
                else:
                    try:
                        validate_pcd_ext(ext)
                        pcd_list.append(full_path)
                    except:
                        continue

        # create Items
        self._items = []
        project = {}
        sly_ann_detected = False
        for pcd_path in pcd_list:
            name_noext = get_file_name(pcd_path)
            item = self.Item(pcd_path)
            rimg_dir_name = item.name.replace(".", "_")
            ds_dir = helpers.get_dataset_dir(pcd_path)
            if ds_dir is not None:
                # project layout: <ds>/pointcloud/<name> -> <ds>/ann/<name>.json and
                # <ds>/related_images/<name with "." -> "_">/, never another dataset's files
                ann_dir = os.path.join(ds_dir, "ann")
                ann_path = os.path.join(ann_dir, f"{item.name}.json")
                if not os.path.isfile(ann_path):
                    ann_path = os.path.join(ann_dir, f"{name_noext}.json")
                ann_path = ann_path if os.path.isfile(ann_path) else None
                rimg_dir = os.path.join(ds_dir, "related_images", rimg_dir_name)
                rimgs = rimg_dict.get(rimg_dir, [])
            else:
                # files in any other layout are matched by name, as before
                ann_name = f"{item.name}.json"
                if ann_name not in ann_dict:
                    ann_name = f"{name_noext}.json"
                ann_path = ann_dict.get(ann_name)
                rimgs = [
                    path
                    for rimg_dir, paths in rimg_dict.items()
                    if os.path.basename(rimg_dir) == rimg_dir_name
                    for path in paths
                ]
            if ann_path is not None:
                if self._meta is None:
                    self._meta = self.generate_meta_from_annotation(ann_path, self._meta)
                is_valid = self.validate_ann_file(ann_path, self._meta)
                if is_valid:
                    item.ann_data = ann_path
                    sly_ann_detected = True

            for rimg_path in rimgs:
                rimg_name = get_file_name_with_ext(rimg_path)
                if ds_dir is not None:
                    rimg_ann_path = os.path.join(os.path.dirname(rimg_path), f"{rimg_name}.json")
                    rimg_fig_path = os.path.join(
                        os.path.dirname(rimg_path), f"{rimg_name}.figures.json"
                    )
                    if not os.path.isfile(rimg_ann_path):
                        continue
                else:
                    if f"{rimg_name}.json" not in rimg_ann_dict:
                        continue
                    rimg_ann_path = rimg_ann_dict[f"{rimg_name}.json"]
                    rimg_fig_path = rimg_ann_dict.get(f"{rimg_name}.figures.json", None)
                if rimg_fig_path is not None and not os.path.exists(rimg_fig_path):
                    rimg_fig_path = None
                item.set_related_images((rimg_path, rimg_ann_path, rimg_fig_path))
            self._items.append(item)
            if project is not None and ds_dir is not None:
                node = helpers.add_dataset_node(project, helpers.get_dataset_name(ds_dir))
                node[DATASET_ITEMS].append(item)
            else:
                project = None  # some clouds are not in the project layout: one dataset

        if project is not None and helpers.count_datasets(project) > 1:
            self._project_structure = project
        return sly_ann_detected

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 1,
        log_progress=True,
    ) -> None:
        if self._project_structure:
            self.upload_project(api, dataset_id, batch_size, log_progress)
        else:
            super().upload_dataset(api, dataset_id, batch_size, log_progress)

    def upload_project(
        self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True
    ) -> None:
        """Upload a project with several datasets, keeping the dataset tree.

        The first top-level dataset goes into the dataset ``dataset_id``, which is renamed after
        it; the others are created next to it, as for images, videos and meshes.
        """
        dataset_info = api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_id = dataset_info.project_id
        existing_datasets = {ds.name for ds in api.dataset.get_list(project_id, recursive=True)}

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading pointclouds...")
        else:
            progress, progress_cb = None, None

        def _upload_project(
            project_structure: Dict,
            project_id: int,
            dataset_id: int,
            parent_id: Optional[int] = None,
            first_dataset: bool = False,
        ):
            for ds_name, value in project_structure.items():
                ds_name = generate_free_name(existing_datasets, ds_name, extend_used_names=True)
                if first_dataset:
                    first_dataset = False
                    api.dataset.update(dataset_id, ds_name)  # rename the pre-created dataset
                else:
                    dataset_id = api.dataset.create(project_id, ds_name, parent_id=parent_id).id

                items = value.get(DATASET_ITEMS, [])
                nested = value.get(NESTED_DATASETS, {})
                logger.info(
                    f"Dataset: {ds_name}, items: {len(items)}, nested datasets: {len(nested)}"
                )
                if items:
                    super(SLYPointcloudConverter, self).upload_dataset(
                        api,
                        dataset_id,
                        batch_size,
                        log_progress=False,
                        entities=items,
                        progress_cb=progress_cb,
                    )
                if nested:
                    _upload_project(nested, project_id, dataset_id, dataset_id)

        _upload_project(self._project_structure, project_id, dataset_id, first_dataset=True)

        if is_development() and progress is not None:
            progress.close()

    def to_supervisely(
        self,
        item: PointcloudConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> PointcloudAnnotation:
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
                ann_json = helpers.rename_in_json(ann_json, renamed_classes, renamed_tags)
            return PointcloudAnnotation.from_json(ann_json, meta)
        except Exception as e:
            logger.warning(f"Failed to convert annotation: {repr(e)}")
            return item.create_empty_annotation()
