import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

from supervisely.api.api import Api
import supervisely.convert.video.sly.sly_video_helper as sly_video_helper
from supervisely import OpenMode, ProjectMeta, VideoAnnotation, VideoProject, logger
from supervisely.convert.base_converter import AvailableVideoConverters
from supervisely.convert.video.video_converter import VideoConverter
from supervisely.io.fs import JUNK_FILES, file_exists, get_file_ext
from supervisely.io.json import load_json_file
from supervisely.project.project import find_project_dirs
from supervisely.project.project_settings import LabelingInterface
from supervisely.video.video import has_valid_ext, validate_ext

DATASET_ITEMS = "items"
NESTED_DATASETS = "datasets"


class MultiViewVideoConverter(VideoConverter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._supports_links = True
        self._project_structure = None

    def __str__(self) -> str:
        return AvailableVideoConverters.MULTI_VIEW

    @property
    def ann_ext(self) -> str:
        return ".json"

    @property
    def key_file_ext(self) -> str:
        return ".json"

    @staticmethod
    def _create_project_node() -> Dict[str, dict]:
        return {DATASET_ITEMS: [], NESTED_DATASETS: {}}

    @classmethod
    def _append_to_project_structure(
        cls, project_structure: Dict[str, dict], dataset_name: str, items: list
    ):
        normalized_name = (dataset_name or "").replace("\\", "/").strip("/")
        if not normalized_name:
            normalized_name = dataset_name or "dataset"
        parts = [part for part in normalized_name.split("/") if part]
        if not parts:
            parts = ["dataset"]

        curr_ds = project_structure.setdefault(parts[0], cls._create_project_node())
        for part in parts[1:]:
            curr_ds = curr_ds[NESTED_DATASETS].setdefault(part, cls._create_project_node())
        curr_ds[DATASET_ITEMS].extend(items)

    def validate_labeling_interface(self) -> bool:
        return self._labeling_interface == LabelingInterface.MULTIVIEW

    def generate_meta_from_annotation(self, ann_path: str, meta: ProjectMeta) -> ProjectMeta:
        meta = sly_video_helper.get_meta_from_annotation(ann_path, meta)
        return meta

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta) -> bool:
        try:
            ann_json = load_json_file(ann_path)
            if "annotation" in ann_json:
                ann_json = ann_json["annotation"]
            VideoAnnotation.from_json(ann_json, meta)
            return True
        except Exception:
            return False

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            self._meta = ProjectMeta.from_json(load_json_file(key_file_path))
            return True
        except Exception:
            return False

    def read_multiview_project(self, input_data: str) -> bool:
        """Read multi-view video project with multiple datasets."""
        try:
            self._items = []
            project = {}
            ds_cnt = 0
            self._meta = None

            logger.debug("Trying to find Supervisely video project format in the input data")
            project_dirs = [d for d in find_project_dirs(input_data, project_class=VideoProject)]
            if len(project_dirs) > 1:
                logger.info("Found multiple possible Supervisely video projects in the input data")
            elif len(project_dirs) == 1:
                logger.info("Possible Supervisely video project found in the input data")
            else:
                return False

            meta = None
            for project_dir in project_dirs:
                project_fs = VideoProject(project_dir, mode=OpenMode.READ)
                if meta is None:
                    meta = project_fs.meta
                else:
                    meta = meta.merge(project_fs.meta)

                for dataset in project_fs.datasets:
                    ds_items = []
                    for name in dataset.get_items_names():
                        video_path, ann_path = dataset.get_item_paths(name)
                        metadata_path = os.path.join(
                            dataset.directory, "metadata", f"{name}.meta.json"
                        )
                        item = self.Item(video_path)
                        if file_exists(ann_path):
                            if self.validate_ann_file(ann_path, meta):
                                item.ann_data = ann_path
                        if file_exists(metadata_path):
                            item.metadata = metadata_path
                        ds_items.append(item)

                    if len(ds_items) > 0:
                        self._append_to_project_structure(project, dataset.name, ds_items)
                        ds_cnt += 1
                        self._items.extend(ds_items)

            if self.items_count > 0:
                self._meta = meta
                if ds_cnt > 1:
                    self._project_structure = project
                return True
            else:
                return False
        except Exception as e:
            logger.debug(f"Not a multi-view video project: {repr(e)}")
            return False

    def read_multiview_dataset(self, input_data: str) -> bool:
        """Read multi-view video datasets without project meta.json."""
        try:
            from supervisely import VideoDataset
            from supervisely.io.fs import dirs_filter

            self._items = []
            project = {}
            ds_cnt = 0
            self._meta = None
            logger.debug("Trying to read Supervisely video datasets")

            def _check_function(path):
                try:
                    dataset_ds = VideoDataset(path, OpenMode.READ)
                    return len(dataset_ds.get_items_names()) > 0
                except:
                    return False

            meta = ProjectMeta()
            dataset_dirs = [d for d in dirs_filter(input_data, _check_function)]
            for dataset_dir in dataset_dirs:
                dataset_fs = VideoDataset(dataset_dir, OpenMode.READ)
                ds_items = []
                for name in dataset_fs.get_items_names():
                    video_path, ann_path = dataset_fs.get_item_paths(name)
                    metadata_path = os.path.join(
                        dataset_fs.directory, "metadata", f"{name}.meta.json"
                    )

                    item = self.Item(video_path)
                    if file_exists(ann_path):
                        meta = self.generate_meta_from_annotation(ann_path, meta)
                        if self.validate_ann_file(ann_path, meta):
                            item.ann_data = ann_path
                    if file_exists(metadata_path):
                        item.metadata = metadata_path
                    ds_items.append(item)

                if len(ds_items) > 0:
                    self._append_to_project_structure(project, dataset_fs.name, ds_items)
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
            logger.debug(f"Failed to read Supervisely video datasets: {repr(e)}")
            return False

    def read_multiview_folder_structure(self, input_data: str) -> bool:
        """Read multi-view folder layout: <dataset>/video (+optional ann, metadata)."""
        try:
            logger.debug("Trying to read folder-based multi-view structure")
            self._items = []
            project = {}
            ds_cnt = 0
            self._meta = None
            self._project_structure = None

            has_meta_file = False
            for file in Path(input_data).rglob("meta.json"):
                if file.is_file() and self.validate_key_file(str(file)):
                    has_meta_file = True
                    break
            meta = self._meta if has_meta_file else ProjectMeta()

            video_groups = self._find_video_groups(input_data)

            for dataset_name, video_paths in video_groups.items():
                ds_items = []
                for path in video_paths:
                    item = self.Item(path)

                    # check both levels
                    possible_ann_dirs = [Path(path).parent.parent, Path(path).parent]

                    for possible_dir in possible_ann_dirs:
                        ann_path = possible_dir / "ann" / f"{item.name}.json"
                        if not ann_path.exists():
                            ann_path = possible_dir / f"{item.name}.json"
                        if ann_path.exists():
                            if not has_meta_file:
                                meta = self.generate_meta_from_annotation(str(ann_path), meta)
                            if self.validate_ann_file(str(ann_path), meta):
                                item.ann_data = str(ann_path)

                        item_meta = possible_dir / "metadata" / f"{item.name}.meta.json"
                        if not item_meta.exists():
                            item_meta = possible_dir / f"{item.name}.meta.json"
                        if item_meta.exists():
                            item.metadata = str(item_meta)
                    ds_items.append(item)

                if len(ds_items) == 0:
                    continue

                self._items.extend(ds_items)
                ds_cnt += 1

                self._append_to_project_structure(project, dataset_name, ds_items)

            if self.items_count > 0:
                self._meta = meta
                if ds_cnt > 1:
                    self._project_structure = project
                return True
            else:
                return False
        except Exception as e:
            logger.debug(f"Failed to read folder-based multi-view structure: {repr(e)}")
            return False

    def _find_video_groups(self, path) -> Dict[str, List[str]]:
        video_groups = defaultdict(list)
        for file_path in Path(path).rglob("*"):
            if file_path.is_file() and has_valid_ext(str(file_path)):
                video_groups[file_path.parent].append(str(file_path))

        sanitized = self._sanitize_dataset_names(video_groups)
        return {sanitized[parent]: files for parent, files in video_groups.items()}

    def _sanitize_dataset_names(self, video_groups: Dict[Path, list]) -> Dict[Path, str]:
        name_counts = defaultdict(int)
        sanitized = {}
        for parent in video_groups:
            base_name = parent.name or "root"
            name_counts[base_name] += 1
            count = name_counts[base_name]
            sanitized[parent] = base_name if count == 1 else f"{base_name}_{count}"
        return sanitized

    def validate_format(self) -> bool:
        if self.upload_as_links and self._supports_links:
            self._download_remote_ann_files()

        if self.read_multiview_project(self._input_data):
            return True

        if self.read_multiview_dataset(self._input_data):
            return True

        if self.read_multiview_folder_structure(self._input_data):
            return True

        detected_ann_cnt = 0
        videos_list, ann_dict, meta_dict = [], {}, {}
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                if file == "key_id_map.json":
                    continue
                if file == "meta.json":
                    is_valid = self.validate_key_file(full_path)
                    if is_valid:
                        continue

                ext = get_file_ext(full_path)
                if file in JUNK_FILES:
                    continue

                elif file.endswith(".meta.json"):
                    meta_dict[file] = full_path
                elif ext in self.ann_ext:
                    ann_dict[file] = full_path
                else:
                    try:
                        validate_ext(ext)  # validate video extension
                        videos_list.append(full_path)
                    except Exception:
                        continue

        if self._meta is not None:
            meta = self._meta
        else:
            meta = ProjectMeta()

        self._items = []
        for video_path in videos_list:
            item = self.Item(video_path)
            ann_name = f"{item.name}.json"
            if ann_name in ann_dict:
                ann_path = ann_dict[ann_name]
                if self._meta is None:
                    meta = self.generate_meta_from_annotation(ann_path, meta)
                is_valid = self.validate_ann_file(ann_path, meta)
                if is_valid:
                    item.ann_data = ann_path
                    detected_ann_cnt += 1

            meta_name = f"{item.name}.meta.json"
            if meta_name in meta_dict:
                meta_path = meta_dict[meta_name]
                item.metadata = meta_path

            self._items.append(item)
        self._meta = meta
        return len(self._items) > 0

    def upload_dataset(self, api, dataset_id: int, batch_size: int = 10, log_progress=True):
        """Upload converted data to Supervisely."""
        if self._project_structure:
            self._upload_project(api, dataset_id, batch_size, log_progress)
        else:
            self._upload_single_dataset(api, dataset_id, self._items, batch_size, log_progress)

    def _upload_project(self, api, dataset_id: int, batch_size: int = 10, log_progress=True):
        """Upload multi-view video project with multiple datasets."""
        from supervisely import generate_free_name, is_development

        dataset_info = api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_id = dataset_info.project_id
        existing_datasets = api.dataset.get_list(project_id, recursive=True)
        existing_datasets = {ds.name for ds in existing_datasets}

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading project")
        else:
            progress, progress_cb = None, None

        logger.info("Uploading multi-view video project structure")

        def _upload_datasets_recursive(
            project_structure: dict,
            project_id: int,
            dataset_id: int,
            parent_id=None,
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
                    self._upload_single_dataset(
                        api,
                        dataset_id,
                        items,
                        batch_size,
                        log_progress=False,
                        progress_cb=progress_cb,
                    )

                if nested_datasets:
                    _upload_datasets_recursive(nested_datasets, project_id, dataset_id, dataset_id)

        _upload_datasets_recursive(
            self._project_structure, project_id, dataset_id, first_dataset=True
        )

        if is_development() and progress is not None:
            progress.close()

    def _upload_single_dataset(
        self,
        api: Api,
        dataset_id: int,
        items: list,
        batch_size: int = 10,
        log_progress=True,
        progress_cb=None,
    ):
        """Upload videos from a single dataset."""
        from supervisely import batched, generate_free_name, is_development
        from supervisely.io.fs import get_file_size
        from supervisely.io.json import load_json_file

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)
        videos_in_dataset = api.video.get_list(dataset_id, force_metadata_for_links=False)
        existing_names = {video_info.name for video_info in videos_in_dataset}
        items_count = len(items)
        convert_progress, convert_progress_cb = self.get_progress(
            items_count, "Preparing videos..."
        )
        for item in items:
            item_name, item_path = self.convert_to_mp4_if_needed(item.path)
            item.name = item_name
            item.path = item_path
            convert_progress_cb(1)
        if is_development():
            convert_progress.close()

        has_large_files = False
        size_progress_cb = None
        _progress_cb, progress, ann_progress, ann_progress_cb = None, None, None, None
        if log_progress:
            if progress_cb is None:
                progress, _progress_cb = self.get_progress(items_count, "Uploading videos...")
            else:
                _progress_cb = progress_cb
            if not self.upload_as_links:
                file_sizes = [get_file_size(item.path) for item in items]
                has_large_files = any(
                    [self._check_video_file_size(file_size) for file_size in file_sizes]
                )
                if has_large_files:
                    upload_progress = []
                    size_progress_cb = self._get_video_upload_progress(upload_progress)

        batch_size = 1 if has_large_files and not self.upload_as_links else batch_size
        for batch in batched(items, batch_size=batch_size):
            item_names = []
            item_paths = []
            item_metas = []
            anns = []
            figures_cnt = 0
            for item in batch:
                item.name = generate_free_name(
                    existing_names, item.name, with_ext=True, extend_used_names=True
                )
                item_paths.append(item.path)
                item_names.append(item.name)

                if isinstance(item.metadata, str):  # path to file
                    item_metas.append(load_json_file(item.metadata))
                elif isinstance(item.metadata, dict):
                    item_metas.append(item.metadata)
                else:
                    item_metas.append({})

                ann = None
                if not self.upload_as_links or self.supports_links:
                    ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
                    if ann is not None:
                        figures_cnt += len(ann.figures)
                anns.append(ann)

            if self.upload_as_links:
                vid_infos = api.video.upload_links(
                    dataset_id,
                    item_paths,
                    item_names,
                    metas=item_metas,
                    skip_download=True,
                    progress_cb=_progress_cb if log_progress else None,
                    force_metadata_for_links=False,
                )
            else:
                vid_infos = api.video.upload_paths(
                    dataset_id,
                    item_names,
                    item_paths,
                    metas=item_metas,
                    progress_cb=_progress_cb if log_progress else None,
                    item_progress=(size_progress_cb if log_progress and has_large_files else None),
                )

            vid_ids = [vid_info.id for vid_info in vid_infos]
            if log_progress and has_large_files and figures_cnt > 0:
                ann_progress, ann_progress_cb = self.get_progress(
                    figures_cnt, "Uploading annotations..."
                )

            for idx, (ann, info) in enumerate(zip(anns, vid_infos)):
                if ann is None:
                    anns[idx] = VideoAnnotation(
                        (info.frame_height, info.frame_width), info.frames_count
                    )
            api.video.annotation.upload_anns_multiview(vid_ids, anns, ann_progress_cb)

        if log_progress and is_development():
            if progress is not None:
                progress.close()
            if ann_progress is not None:
                ann_progress.close()
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")

    def to_supervisely(
        self,
        item: VideoConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> VideoAnnotation:
        if meta is None:
            meta = self._meta

        if item.ann_data is None:
            if self._upload_as_links:
                return None
            return item.create_empty_annotation()

        try:
            ann_json = load_json_file(item.ann_data)
            if "annotation" in ann_json:
                ann_json = ann_json["annotation"]
            if renamed_classes or renamed_tags:
                ann_json = sly_video_helper.rename_in_json(ann_json, renamed_classes, renamed_tags)
            return VideoAnnotation.from_json(ann_json, meta)
        except Exception as e:
            logger.warning(f"Failed to convert annotation: {repr(e)}")
            return item.create_empty_annotation()
