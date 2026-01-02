import imghdr
import os
from typing import List

import supervisely.convert.pointcloud.sly.sly_pointcloud_helper as helpers
from supervisely import PointcloudAnnotation, ProjectMeta, logger
from supervisely.api.api import Api
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.convert.utils import ProjectStructureUploader
from supervisely.io.fs import get_file_ext, get_file_name, get_file_name_with_ext
from supervisely.io.json import load_json_file
from supervisely.pointcloud.pointcloud import validate_ext as validate_pcd_ext

DATASET_ITEMS = "items"
NESTED_DATASETS = "datasets"

_IGNORED_DATASET_PARTS = {
    "datasets",
    "pointcloud",
    "ann",
    "meta",
    "related_images",
}


class SLYPointcloudConverter(PointcloudConverter):

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._effective_input_data = None

    def _get_effective_input_data(self) -> str:
        """
        If input is an extracted archive with a single top-level folder wrapper,
        return that folder as project root to avoid 'archive_name -> project' nesting.
        """
        if self._effective_input_data is not None:
            return self._effective_input_data

        root = os.path.abspath(self._input_data)
        try:
            entries = [e for e in os.listdir(root) if not e.startswith(".")]
            entries = [e for e in entries if e not in {"__MACOSX"}]
            dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
            files = [e for e in entries if os.path.isfile(os.path.join(root, e))]

            if len(files) == 0 and len(dirs) == 1:
                self._effective_input_data = os.path.join(root, dirs[0])
            else:
                self._effective_input_data = root
        except Exception:
            self._effective_input_data = root
        return self._effective_input_data

    def _infer_dataset_name(self, pcd_path: str) -> str:
        p = os.path.dirname(os.path.abspath(pcd_path))
        root = os.path.abspath(self._get_effective_input_data())

        def _norm(name: str) -> str:
            return name.strip().lower().replace("_", " ")

        while True:
            base = os.path.basename(p)
            if _norm(base) in _IGNORED_DATASET_PARTS:
                parent = os.path.dirname(p)
                if parent == p:
                    break
                p = parent
                continue
            break

        rel_dir = os.path.relpath(p, root)
        if rel_dir in (".", ""):
            return "dataset"

        parts = [seg for seg in rel_dir.replace("\\", "/").split("/") if seg and seg != "."]

        cleaned = [seg for seg in parts if _norm(seg) not in _IGNORED_DATASET_PARTS]

        return "/".join(cleaned) if cleaned else "dataset"

    def validate_format(self) -> bool:
        input_root = self._get_effective_input_data()

        pcd_list, ann_dict, rimg_dict, rimg_ann_dict = [], {}, {}, {}
        for root, _, files in os.walk(input_root):
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
                elif imghdr.what(full_path):
                    if dir_name not in rimg_dict:
                        rimg_dict[dir_name] = []
                    rimg_dict[dir_name].append(full_path)
                else:
                    try:
                        validate_pcd_ext(ext)
                        pcd_list.append(full_path)
                    except:
                        continue

        # create Items
        self._items = []
        project = {}
        dataset_names_seen = set()
        sly_ann_detected = False
        for pcd_path in pcd_list:
            name_noext = get_file_name(pcd_path)
            item = self.Item(pcd_path)
            ann_name = f"{item.name}.json"
            if ann_name not in ann_dict:
                ann_name = f"{name_noext}.json"
            if ann_name in ann_dict:
                ann_path = ann_dict[ann_name]
                if self._meta is None:
                    self._meta = self.generate_meta_from_annotation(ann_path, self._meta)
                is_valid = self.validate_ann_file(ann_path, self._meta)
                if is_valid:
                    item.ann_data = ann_path
                    sly_ann_detected = True

            rimg_dir_name = item.name.replace(".pcd", "_pcd")
            rimgs = rimg_dict.get(rimg_dir_name, [])
            for rimg_path in rimgs:
                rimg_ann_name = f"{get_file_name_with_ext(rimg_path)}.json"
                if rimg_ann_name in rimg_ann_dict:
                    rimg_ann_path = rimg_ann_dict[rimg_ann_name]
                    rimg_fig_name = f"{get_file_name_with_ext(rimg_path)}.figures.json"
                    rimg_fig_path = rimg_ann_dict.get(rimg_fig_name, None)
                    if rimg_fig_path is not None and not os.path.exists(rimg_fig_path):
                        rimg_fig_path = None
                    item.set_related_images((rimg_path, rimg_ann_path, rimg_fig_path))
            self._items.append(item)

            ds_name = self._infer_dataset_name(pcd_path)
            dataset_names_seen.add(ds_name)
            ProjectStructureUploader.append_items(project, ds_name, [item])

        self._project_structure = project if len(dataset_names_seen) > 1 else None
        return sly_ann_detected

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

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 10, log_progress=True):
        if getattr(self, "_project_structure", None):
            self._upload_project(api, dataset_id, batch_size, log_progress)
        else:
            return super().upload_dataset(
                api, dataset_id, batch_size=batch_size, log_progress=log_progress
            )

    def _upload_project(self, api: Api, dataset_id: int, batch_size: int = 10, log_progress=True):
        from supervisely import is_development

        dataset_info = api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_id = dataset_info.project_id
        existing_datasets = api.dataset.get_list(project_id, recursive=True)
        existing_datasets = {ds.name for ds in existing_datasets}

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading project")
        else:
            progress, progress_cb = None, None

        def _upload_single_dataset_cb(_dataset_id: int, items: list):
            try:
                return super().upload_dataset(
                    api, _dataset_id, batch_size=batch_size, log_progress=False
                )
            except Exception:
                meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(
                    api, _dataset_id
                )
                for it in items:
                    pc_info = api.pointcloud.upload_path(_dataset_id, it.name, it.path)
                    ann = self.to_supervisely(it, meta, renamed_classes, renamed_tags)
                    api.pointcloud.annotation.append(pc_info.id, ann)
                    if progress_cb:
                        progress_cb(1)

        ProjectStructureUploader(existing_datasets=existing_datasets).upload(
            api=api,
            project_id=project_id,
            root_dataset_id=dataset_id,
            project_structure=self._project_structure,
            upload_items_cb=_upload_single_dataset_cb,
        )

        if is_development() and progress is not None:
            progress.close()
