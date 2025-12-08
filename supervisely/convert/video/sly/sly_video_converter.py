import os
from typing import Dict, List, Optional

import supervisely.convert.video.sly.sly_video_helper as sly_video_helper
from supervisely import ProjectMeta, VideoAnnotation, logger
from supervisely.api.api import Api, ApiContext
from supervisely.convert.base_converter import AvailableVideoConverters
from supervisely.convert.video.video_converter import VideoConverter
from supervisely.io.fs import JUNK_FILES, file_exists, get_file_ext
from supervisely.io.json import load_json_file
from supervisely.project.project import OpenMode, find_project_dirs
from supervisely.project.project_settings import LabelingInterface
from supervisely.project.video_project import VideoProject
from supervisely.video.video import validate_ext as validate_video_ext

DATASET_ITEMS = "items"
NESTED_DATASETS = "datasets"


class SLYVideoConverter(VideoConverter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._supports_links = True
        self._project_structure = None
        self._multi_view_setting_enabled = False

    def __str__(self) -> str:
        return AvailableVideoConverters.SLY

    @property
    def ann_ext(self) -> str:
        return ".json"

    @property
    def key_file_ext(self) -> str:
        return ".json"

    def generate_meta_from_annotation(self, ann_path: str, meta: ProjectMeta) -> ProjectMeta:
        meta = sly_video_helper.get_meta_from_annotation(ann_path, meta)
        return meta

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta) -> bool:
        try:
            ann_json = load_json_file(ann_path)
            if "annotation" in ann_json:
                ann_json = ann_json["annotation"]
            ann = VideoAnnotation.from_json(ann_json, meta)
            return True
        except:
            return False

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            self._meta = ProjectMeta.from_json(load_json_file(key_file_path))
            return True
        except Exception:
            return False

    @staticmethod
    def _create_project_node() -> Dict[str, dict]:
        return {DATASET_ITEMS: [], NESTED_DATASETS: {}}

    @classmethod
    def _append_to_project_structure(
        cls, project_structure: Dict[str, dict], dataset_name: str, items: List
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

    def validate_format(self) -> bool:
        if self.upload_as_links and self._supports_links:
            self._download_remote_ann_files()
        if self.read_project_structure(self._input_data):
            return True

        if self.read_dataset_structure(self._input_data):
            return True
        detected_ann_cnt = 0
        videos_list, ann_dict = [], {}
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
                if file in JUNK_FILES:  # add better check
                    continue
                elif ext in self.ann_ext:
                    ann_dict[file] = full_path
                else:
                    try:
                        validate_video_ext(ext)
                        videos_list.append(full_path)
                    except:
                        continue

        if self._meta is not None:
            meta = self._meta
        else:
            meta = ProjectMeta()

        # create Items
        self._items = []
        for image_path in videos_list:
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
            self._items.append(item)
        self._meta = meta
        return detected_ann_cnt > 0

    def to_supervisely(
        self,
        item: VideoConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> VideoAnnotation:
        """Convert to Supervisely format."""
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

    def read_project_structure(self, input_data: str) -> bool:
        """Read video project with multiple datasets."""
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
                        item = self.Item(video_path)
                        if file_exists(ann_path):
                            if self.validate_ann_file(ann_path, meta):
                                item.ann_data = ann_path
                        ds_items.append(item)

                    if len(ds_items) > 0:
                        self._append_to_project_structure(project, dataset.name, ds_items)
                        ds_cnt += 1
                        self._items.extend(ds_items)

            if self.items_count > 0:
                self._meta = meta
                meta: ProjectMeta
                if meta.labeling_interface == LabelingInterface.MULTIVIEW:
                    self._multi_view_setting_enabled = True
                if ds_cnt > 1:
                    self._project_structure = project
                return True
            else:
                return False
        except Exception as e:
            logger.debug(f"Not a video project: {repr(e)}")
            return False

    def read_dataset_structure(self, input_data: str) -> bool:
        """Read video datasets without project meta.json."""
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
                except Exception:
                    return False

            meta = ProjectMeta()
            dataset_dirs = [d for d in dirs_filter(input_data, _check_function)]
            for dataset_dir in dataset_dirs:
                dataset_fs = VideoDataset(dataset_dir, OpenMode.READ)
                ds_items = []
                for name in dataset_fs.get_items_names():
                    video_path, ann_path = dataset_fs.get_item_paths(name)
                    item = self.Item(video_path)
                    if file_exists(ann_path):
                        meta = self.generate_meta_from_annotation(ann_path, meta)
                        if self.validate_ann_file(ann_path, meta):
                            item.ann_data = ann_path
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

    def upload_dataset(
        self, api: Api, dataset_id: int, batch_size: int = 10, log_progress=True
    ) -> Optional[int]:
        """Upload converted data to Supervisely."""
        if self._project_structure:
            return self._upload_project(api, dataset_id, batch_size, log_progress)
        else:
            self._upload_single_dataset(api, dataset_id, self._items, batch_size, log_progress)

    def _upload_project(
        self, api: Api, dataset_id: int, batch_size: int = 10, log_progress=True
    ) -> Optional[int]:
        """Upload video project with multiple datasets."""
        from supervisely import generate_free_name, is_development

        dataset_info = api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_id = dataset_info.project_id
        new_project_created = False

        if self._multi_view_setting_enabled:
            src_meta_json = api.project.get_meta(project_id, with_settings=True)
            src_meta = ProjectMeta.from_json(src_meta_json)

            if src_meta.labeling_interface == LabelingInterface.DEFAULT:
                project_id, dataset_id = self._handle_multi_view_labeling_interface(
                    api, project_id, dataset_info
                )
                new_project_created = True

        existing_datasets = api.dataset.get_list(project_id, recursive=True)
        existing_datasets = {ds.name for ds in existing_datasets}

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading project")
        else:
            progress, progress_cb = None, None

        logger.info("Uploading video project structure")

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

        if new_project_created:
            logger.info(
                "Data was uploaded to a new project with 'Multi-View' labeling interface setting."
            )
            return dataset_id

    def _upload_single_dataset(
        self,
        api: Api,
        dataset_id: int,
        items: List,
        batch_size: int = 10,
        log_progress=True,
        progress_cb=None,
    ):
        """Upload videos from a single dataset."""
        from supervisely import batched, generate_free_name, is_development
        from supervisely.io.fs import get_file_size

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

        with ApiContext(api=api, project_meta=meta):
            batch_size = 1 if has_large_files and not self.upload_as_links else batch_size
            for batch in batched(items, batch_size=batch_size):
                item_names = []
                item_paths = []
                anns = []
                figures_cnt = 0
                for item in batch:
                    item.name = generate_free_name(
                        existing_names, item.name, with_ext=True, extend_used_names=True
                    )
                    item_paths.append(item.path)
                    item_names.append(item.name)

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
                        skip_download=True,
                        progress_cb=_progress_cb if log_progress else None,
                        force_metadata_for_links=False,
                    )
                else:
                    vid_infos = api.video.upload_paths(
                        dataset_id,
                        item_names,
                        item_paths,
                        progress_cb=_progress_cb if log_progress else None,
                        item_progress=(
                            size_progress_cb if log_progress and has_large_files else None
                        ),
                    )

                vid_ids = [vid_info.id for vid_info in vid_infos]
                if log_progress and has_large_files and figures_cnt > 0:
                    ann_progress, ann_progress_cb = self.get_progress(
                        figures_cnt, "Uploading annotations..."
                    )

                if meta.labeling_interface == LabelingInterface.MULTIVIEW:
                    for idx, (ann, info) in enumerate(zip(anns, vid_infos)):
                        if ann is None:
                            anns[idx] = VideoAnnotation(
                                (info.frame_height, info.frame_width), info.frames_count
                            )
                    api.video.annotation.upload_anns_multiview(vid_ids, anns, ann_progress_cb)
                else:
                    for vid, ann, info in zip(vid_ids, anns, vid_infos):
                        if ann is None:
                            ann = VideoAnnotation(
                                (info.frame_height, info.frame_width), info.frames_count
                            )
                        api.video.annotation.append(vid, ann, progress_cb=ann_progress_cb)

        if log_progress and is_development():
            if progress is not None:
                progress.close()
            if ann_progress is not None:
                ann_progress.close()
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")

    def _handle_multi_view_labeling_interface(self, api: Api, project_id: int, dataset_info):
        project_info = api.project.get_info_by_id(project_id)
        if project_info.items_count == 0:
            return project_id, dataset_info.id
        logger.warning(
            "The uploaded project has 'Multi-View' labeling interface setting enabled, "
            "but the target project has 'Default' labeling interface. "
        )
        logger.warning("New project with 'Multi-View' labeling interface will be created.")
        new_project = api.project.create(
            workspace_id=project_info.workspace_id,
            name=f"{project_info.name}_multi_view",
            type=project_info.type,
            change_name_if_conflict=True,
        )
        new_dataset = api.dataset.create(
            new_project.id, dataset_info.name, change_name_if_conflict=True
        )
        return new_project.id, new_dataset.id
