import imghdr
import os

import supervisely.convert.pointcloud_episodes.sly.sly_pointcloud_episodes_helper as sly_episodes_helper
from supervisely import PointcloudEpisodeAnnotation, ProjectMeta, logger
from supervisely.convert.base_converter import AvailablePointcloudEpisodesConverters
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import (
    PointcloudEpisodeConverter,
)
from supervisely.convert.utils import ProjectStructureUploader
from supervisely.io.fs import JUNK_FILES, get_file_ext, get_file_name_with_ext
from supervisely.io.json import load_json_file
from supervisely.pointcloud.pointcloud import validate_ext as validate_pcd_ext

_IGNORED_DATASET_PARTS = {
    "datasets",
    "pointcloud",
    "ann",
    "meta",
    "related_images",
}


class SLYPointcloudEpisodesConverter(PointcloudEpisodeConverter):
    def __str__(self) -> str:
        return AvailablePointcloudEpisodesConverters.SLY

    @property
    def ann_ext(self) -> str:
        return ".json"

    @property
    def key_file_ext(self) -> str:
        return ".json"

    def generate_meta_from_annotation(self, ann_path: str, meta: ProjectMeta) -> ProjectMeta:
        meta = sly_episodes_helper.get_meta_from_annotation(ann_path, meta)
        return meta

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta) -> bool:
        try:
            ann_json = load_json_file(ann_path)
            if "annotation" in ann_json:
                ann_json = ann_json["annotation"]
            ann = PointcloudEpisodeAnnotation.from_json(ann_json, meta)
            self._annotation = ann
            return True
        except Exception as e:
            return False

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            self._meta = ProjectMeta.from_json(load_json_file(key_file_path))
            return True
        except Exception:
            return False

    def _infer_dataset_name(self, pcd_path: str) -> str:
        """
        Turns pointcloud path into a full dataset name. Defaults to 'dataset' if no subfolder structure is found.
        """
        p = os.path.dirname(os.path.abspath(pcd_path))
        root = os.path.abspath(self._input_data)

        while True:
            if os.path.basename(p) in _IGNORED_DATASET_PARTS:
                parent = os.path.dirname(p)
                if parent == p:
                    break
                p = parent
                continue
            break

        rel_dir = os.path.relpath(p, root).replace(".", "")

        dataset_names = []
        for part in rel_dir.split(os.sep):
            if part and part not in _IGNORED_DATASET_PARTS:
                dataset_names.append(part)

        return "/".join(dataset_names) if dataset_names else "dataset"

    def validate_format(self) -> bool:
        input_root = self._input_data

        sly_ann_detected = False
        ann_path = None
        pcd_dict = {}
        frames_pcd_map = None
        rimg_dict, rimg_json_dict = {}, {}
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
                if file == "annotation.json":
                    ann_path = full_path
                    continue
                if file == "frame_pointcloud_map.json":
                    frames_pcd_map = load_json_file(full_path)
                    continue

                ext = get_file_ext(full_path)
                recognized_ext = imghdr.what(full_path)
                if file in JUNK_FILES:
                    continue
                elif ext == self.ann_ext:
                    rimg_json_dict[file] = full_path
                elif recognized_ext:
                    if ext.lower() == ".pcd":
                        logger.warning(
                            f"File '{file}' has been recognized as '.{recognized_ext}' format. Skipping."
                        )
                        continue
                    if dir_name not in rimg_dict:
                        rimg_dict[dir_name] = []
                    rimg_dict[dir_name].append(full_path)
                else:
                    try:
                        validate_pcd_ext(ext)
                        pcd_dict[file] = full_path
                    except:
                        continue

        if self._meta is not None:
            meta = self._meta
            sly_ann_detected = True
        else:
            meta = ProjectMeta()
        if ann_path is not None:
            if self._meta is None:
                meta = self.generate_meta_from_annotation(ann_path, meta)
            is_valid = self.validate_ann_file(ann_path, meta)
            if is_valid:
                sly_ann_detected = True

        if not sly_ann_detected:
            return False

        self._items = []
        project = {}
        dataset_names_seen = set()
        updated_frames_pcd_map = {}
        if frames_pcd_map:
            list_of_pcd_names = list(frames_pcd_map.values())
        else:
            list_of_pcd_names = sorted(pcd_dict.keys())

        for i, pcd_name in enumerate(list_of_pcd_names):
            if pcd_name in pcd_dict:
                updated_frames_pcd_map[i] = pcd_name
                item = self.Item(pcd_dict[pcd_name], i)
                rimg_dir_name = pcd_name.replace(".pcd", "_pcd")
                rimgs = rimg_dict.get(rimg_dir_name, [])
                for rimg_path in rimgs:
                    rimg_ann_name = f"{get_file_name_with_ext(rimg_path)}.json"
                    if rimg_ann_name in rimg_json_dict:
                        rimg_ann_path = rimg_json_dict[rimg_ann_name]
                        rimg_fig_name = f"{get_file_name_with_ext(rimg_path)}.figures.json"
                        rimg_fig_path = rimg_json_dict.get(rimg_fig_name, None)
                        if rimg_fig_path is not None and not os.path.exists(rimg_fig_path):
                            rimg_fig_path = None
                        item.set_related_images((rimg_path, rimg_ann_path, rimg_fig_path))
                self._items.append(item)

                ds_name = self._infer_dataset_name(item.path)
                dataset_names_seen.add(ds_name)
                ProjectStructureUploader.append_items(project, ds_name, [item])
            else:
                logger.warning(f"Pointcloud file {pcd_name} not found. Skipping frame.")
                continue

        self._project_structure = project if len(dataset_names_seen) > 1 else None
        self._frame_pointcloud_map = updated_frames_pcd_map
        self._frame_count = len(self._frame_pointcloud_map)

        self._meta = meta
        return sly_ann_detected

    def to_supervisely(
        self,
        item: PointcloudEpisodeConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> PointcloudEpisodeAnnotation:
        """Convert to Supervisely format."""
        if self._annotation is not None:
            if renamed_classes or renamed_tags:
                ann_json = self._annotation.to_json()
                ann_json = sly_episodes_helper.rename_in_json(
                    ann_json, renamed_classes, renamed_tags
                )
                self._annotation = PointcloudEpisodeAnnotation.from_json(ann_json, meta)
            return self._annotation
        else:
            return item.create_empty_annotation()

    def upload_dataset(self, api, dataset_id: int, batch_size: int = 10, log_progress=True):
        """
        Upload converted data to Supervisely.
        Mirrors MultiViewVideoConverter: if _project_structure is present -> create nested datasets.
        """
        if getattr(self, "_project_structure", None):
            self._upload_project(api, dataset_id, batch_size, log_progress)
        else:
            return super().upload_dataset(
                api, dataset_id, batch_size=batch_size, log_progress=log_progress
            )

    def _upload_project(self, api, dataset_id: int, batch_size: int = 10, log_progress=True):
        from supervisely import is_development

        dataset_info = api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_id = dataset_info.project_id
        existing_datasets = api.dataset.get_list(project_id, recursive=True)
        existing_datasets = {ds.name for ds in existing_datasets}

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading project")
        else:
            progress, progress_cb = None, None

        def _upload_single_dataset(_dataset_id: int, items: list):
            # Prefer base implementation for episodes; fallback kept minimal.
            try:
                # Temporarily switch current items/map to only this dataset scope
                prev_items = self._items
                prev_map = getattr(self, "_frame_pointcloud_map", None)
                prev_count = getattr(self, "_frame_count", None)
                try:
                    self._items = items
                    # Rebuild local frame->pcd mapping for this dataset scope
                    self._frame_pointcloud_map = {
                        i: os.path.basename(it.path) for i, it in enumerate(items)
                    }
                    self._frame_count = len(self._frame_pointcloud_map)
                    return super().upload_dataset(
                        api, _dataset_id, batch_size=batch_size, log_progress=False
                    )
                finally:
                    self._items = prev_items
                    self._frame_pointcloud_map = prev_map
                    self._frame_count = prev_count
            except Exception:
                # If base method is unavailable, just advance progress so UI doesn't freeze
                if progress_cb:
                    progress_cb(len(items))

        ProjectStructureUploader(existing_datasets=existing_datasets).upload(
            api=api,
            project_id=project_id,
            root_dataset_id=dataset_id,
            project_structure=self._project_structure,
            upload_items_cb=_upload_single_dataset,
        )

        if is_development() and progress is not None:
            progress.close()
