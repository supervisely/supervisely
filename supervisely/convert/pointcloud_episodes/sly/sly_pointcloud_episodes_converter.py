import os
from typing import Dict, Optional

import supervisely.convert.pointcloud.sly.sly_pointcloud_helper as sly_pointcloud_helper
import supervisely.convert.pointcloud_episodes.sly.sly_pointcloud_episodes_helper as sly_episodes_helper
from supervisely import PointcloudEpisodeAnnotation, ProjectMeta, logger
from supervisely._utils import generate_free_name
from supervisely.api.api import Api
from supervisely.convert.base_converter import AvailablePointcloudEpisodesConverters
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import (
    PointcloudEpisodeConverter,
)
from supervisely.io.fs import JUNK_FILES, get_file_ext, get_file_name_with_ext
from supervisely.io.json import load_json_file
from supervisely.pointcloud.pointcloud import validate_ext as validate_pcd_ext


class SLYPointcloudEpisodesConverter(PointcloudEpisodeConverter):
    """Imports Supervisely point cloud episode project format (PCD sequences + JSON annotations) from local folder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set when the input holds more than one episode (dataset) in the project layout:
        #   { ds_name: { "items": [(items, annotation, frame_pointcloud_map)], "datasets": {...} } }
        self._project_structure = None

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
            logger.warning(f"Failed to parse annotation file '{ann_path}': {e}")
            return False

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            self._meta = ProjectMeta.from_json(load_json_file(key_file_path))
            return True
        except Exception:
            return False

    def validate_format(self) -> bool:
        sly_ann_detected = False
        ann_path = None
        episode_dirs = []
        pcd_dict = {}
        frames_pcd_map = None
        rimg_dict, rimg_json_dict = {}, {}
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
                if file == "annotation.json":
                    ann_path = full_path
                    if os.path.isdir(os.path.join(root, sly_pointcloud_helper.POINTCLOUD_DIR)):
                        episode_dirs.append(root)
                    continue
                if file == "frame_pointcloud_map.json":
                    frames_pcd_map = load_json_file(full_path)
                    continue

                ext = get_file_ext(full_path)
                recognized_ext = self._get_image_subtype(full_path)
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

        if len(episode_dirs) > 1:
            # several episodes in the project layout: each one is a dataset with its own
            # annotation.json, frame_pointcloud_map.json, pointcloud/ and related_images/
            return self._read_episodes(sorted(episode_dirs), meta)

        if not sly_ann_detected:
            return False

        self._items = []
        updated_frames_pcd_map = {}
        if frames_pcd_map:
            list_of_pcd_names = list(frames_pcd_map.values())
        else:
            list_of_pcd_names = sorted(pcd_dict.keys())

        for i, pcd_name in enumerate(list_of_pcd_names):
            if pcd_name in pcd_dict:
                updated_frames_pcd_map[i] = pcd_name
                item = self.Item(pcd_dict[pcd_name], i)
                rimg_dir_name = pcd_name.replace(".", "_")
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
            else:
                logger.warning(f"Pointcloud file {pcd_name} not found. Skipping frame.")
                continue
        self._frame_pointcloud_map = updated_frames_pcd_map
        self._frame_count = len(self._frame_pointcloud_map)

        self._meta = meta
        return sly_ann_detected

    def _read_episodes(self, episode_dirs, meta: ProjectMeta) -> bool:
        project, used_names = {}, set()
        self._items = []
        for ep_dir in episode_dirs:
            ann_path = os.path.join(ep_dir, "annotation.json")
            if self._meta is None:
                meta = self.generate_meta_from_annotation(ann_path, meta)
            self._annotation = None
            if not self.validate_ann_file(ann_path, meta):
                continue
            annotation = self._annotation

            pcd_dir = os.path.join(ep_dir, sly_pointcloud_helper.POINTCLOUD_DIR)
            map_path = os.path.join(ep_dir, "frame_pointcloud_map.json")
            if os.path.isfile(map_path):
                pcd_names = list(load_json_file(map_path).values())
            else:
                pcd_names = sorted(n for n in os.listdir(pcd_dir) if self._is_pcd_name(n))

            items, frame_pcd_map = [], {}
            for frame, pcd_name in enumerate(pcd_names):  # frame numbers as in the single mode
                pcd_path = os.path.join(pcd_dir, pcd_name)
                if not self._is_pcd_name(pcd_name) or not os.path.isfile(pcd_path):
                    logger.warning(f"Pointcloud file {pcd_path} not found. Skipping frame.")
                    continue
                frame_pcd_map[frame] = pcd_name
                item = self.Item(pcd_path, frame)
                rimg_dir = os.path.join(ep_dir, "related_images", pcd_name.replace(".", "_"))
                if os.path.isdir(rimg_dir):
                    for rimg_name in sorted(os.listdir(rimg_dir)):
                        rimg_path = os.path.join(rimg_dir, rimg_name)
                        rimg_ann_path = f"{rimg_path}.json"
                        if not self._get_image_subtype(rimg_path) or not os.path.isfile(
                            rimg_ann_path
                        ):
                            continue
                        rimg_fig_path = f"{rimg_path}.figures.json"
                        if not os.path.isfile(rimg_fig_path):
                            rimg_fig_path = None
                        item.set_related_images((rimg_path, rimg_ann_path, rimg_fig_path))
                items.append(item)

            if items:
                # a dataset holds one episode: the same dataset path twice in the input
                # (two projects) becomes "<name>_01"
                ds_name = sly_pointcloud_helper.get_dataset_name(ep_dir)
                ds_name = generate_free_name(used_names, ds_name, extend_used_names=True)
                node = sly_pointcloud_helper.add_dataset_node(project, ds_name)
                node["items"].append((items, annotation, frame_pcd_map))
                self._items.extend(items)

        self._meta = meta
        if not self._items:
            return False
        if sly_pointcloud_helper.count_datasets(project) > 1:
            self._project_structure = project
        else:  # one valid episode: the single-episode upload
            ((items, annotation, frame_pcd_map),) = list(self._iter_episodes(project))
            self._set_episode(items, annotation, frame_pcd_map)
        return True

    @staticmethod
    def _is_pcd_name(name: str) -> bool:
        try:
            validate_pcd_ext(get_file_ext(name))
            return True
        except Exception:
            return False

    @classmethod
    def _iter_episodes(cls, project: Dict):
        for node in project.values():
            yield from node["items"]
            yield from cls._iter_episodes(node["datasets"])

    def _set_episode(self, items, annotation, frame_pcd_map) -> None:
        self._items = items
        self._annotation = annotation
        self._frame_pointcloud_map = frame_pcd_map
        self._frame_count = len(frame_pcd_map)

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        if not self._project_structure:
            return super().upload_dataset(api, dataset_id, batch_size, log_progress)

        # one dataset per episode; the first top-level one goes into the dataset ``dataset_id``
        # (which is renamed after it), as for other Supervisely-format projects
        dataset_info = api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_id = dataset_info.project_id
        existing_datasets = {ds.name for ds in api.dataset.get_list(project_id, recursive=True)}
        all_items = self._items

        def _upload_project(
            project_structure: Dict,
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
                logger.info(f"Dataset: {ds_name}, episodes: {len(value['items'])}")
                for items, annotation, frame_pcd_map in value["items"]:
                    self._set_episode(items, annotation, frame_pcd_map)
                    super(SLYPointcloudEpisodesConverter, self).upload_dataset(
                        api, dataset_id, batch_size, log_progress
                    )
                if value["datasets"]:
                    _upload_project(value["datasets"], dataset_id, dataset_id)

        try:
            _upload_project(self._project_structure, dataset_id, first_dataset=True)
        finally:
            self._items = all_items

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
