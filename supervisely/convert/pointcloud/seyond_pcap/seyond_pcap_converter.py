import copy
import os
import tempfile
import uuid
from typing import Dict, List, Optional, Tuple, Union

from supervisely import (
    Api,
    PointcloudAnnotation,
    PointcloudEpisodeAnnotation,
    ProjectMeta,
    generate_free_name,
    is_development,
    logger,
)
from supervisely.api.module_api import ApiField
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.convert.pointcloud.seyond_pcap.seyond_pcap_helper import (
    LABELS_DIR_NAME,
    LABELS_FILE_NAME,
    LABELS_META_NAME,
    SENSORS_CONFIG_NAME,
    SensorsConfig,
    sensor_key,
)
from supervisely.convert.pointcloud.sly.sly_pointcloud_helper import rename_in_json
from supervisely.io.fs import (
    JUNK_FILES,
    get_file_ext,
    get_file_hash,
    get_file_name,
    silent_remove,
)
from supervisely.io.json import load_json_file
from supervisely.pointcloud.seyond_pcap import (
    PCAP_EXT,
    SeyondPcapError,
    SeyondPcapIndex,
    index_pcap,
)
from supervisely.project.project_settings import LabelingInterface
from supervisely.video_annotation.key_id_map import KeyIdMap

_CLOUD_STORAGE_PREFIXES = ("s3://", "azure://", "google://", "agent://", "fs://")


class SeyondPcapConverter(PointcloudConverter):
    """Imports Seyond (Innovusion) lidar captures: raw libpcap files of the sensor's UDP output.

    A capture is stored once, by upload, as a link to cloud storage, or by its Team Files id, and
    every lidar frame in it becomes an item that reuses it. The labeling tool decodes each item's
    frame from the file, so nothing is converted and nothing is copied per frame.

    - Each item's ``meta`` holds ``frame``, ``timestamp`` (the capture time of the frame's first
      packet, UTC), ``sensorsData.extrinsicMatrix`` when ``sensors.json`` gives the sensor's mount,
      and ``source``: ``{"file", "frameInFile", "sensor"}``, which the export reads.
    - Point clouds: one dataset per capture, ``meta.frame`` is the frame in the file.
    - Point cloud episodes (:class:`SeyondPcapEpisodesConverter`): the captures of each sensor are
      chained by time into episodes, ``meta.frame`` continues across them.
    - Labels written next to the captures by the export (``supervisely_labels/``) are re-attached.
    """

    class Item(PointcloudConverter.Item):
        """One capture (file) and the frames indexed in it."""

        def __init__(
            self,
            item_path: str,
            index: SeyondPcapIndex,
            source_file: str,
            sensor: str,
            matrix: Optional[List[float]] = None,
            remote_path: Optional[str] = None,
        ):
            super().__init__(item_path)
            self.index = index
            self.source_file = source_file
            self.sensor = sensor
            self.matrix = matrix
            self.remote_path = remote_path

        @property
        def frame_count(self) -> int:
            return self.index.frame_count

        @property
        def start_time(self) -> float:
            return self.index.frames[0].time

        @property
        def end_time(self) -> float:
            return self.index.frames[-1].time

    def __init__(
        self,
        input_data: str,
        labeling_interface: Optional[Union[LabelingInterface, str]],
        upload_as_links: bool,
        remote_files_map: Optional[Dict[str, str]] = None,
        team_files_id_map: Optional[Dict[str, str]] = None,
    ):
        """See :class:`~supervisely.convert.base_converter.BaseConverter` for params."""
        super().__init__(
            input_data, labeling_interface, upload_as_links, remote_files_map, team_files_id_map
        )
        self._supports_links = True
        self._is_pcd_episode = False
        self._config = SensorsConfig()
        # (source file, frame in file) -> frame labels, from a previous export
        self._labels_root: Optional[str] = None
        self._single_labels: Dict[Tuple[str, int], dict] = {}
        self._episode_labels: List[Tuple[dict, Dict[int, Tuple[str, int]]]] = []

    def __str__(self) -> str:
        return AvailablePointcloudConverters.SEYOND_PCAP

    @property
    def key_file_ext(self) -> str:
        return PCAP_EXT

    @property
    def frames_count(self) -> int:
        return sum(item.frame_count for item in self._items)

    def validate_format(self) -> bool:
        pcap_paths, config_path, labels_dir, has_sly_project, other_files = self._scan_input()
        if not pcap_paths or has_sly_project:
            # a Supervisely project with pcap items is the Supervisely converter's
            return False

        if self.upload_as_links:
            self._download_remote_ann_files(exts_to_download=[".json"])

        self._labels_root = os.path.dirname(labels_dir) if labels_dir else None
        root = self._labels_root or self._input_data

        items, skipped, not_seyond = [], [], []
        for path in pcap_paths:
            remote_path = self._remote_path(path)
            try:
                index = self._index_capture(path, remote_path)
            except SeyondPcapError as e:
                not_seyond.append((os.path.basename(path), str(e)))
                continue
            if not index.is_seyond:
                not_seyond.append((os.path.basename(path), index.unsupported_reason()))
                continue
            reason = index.unsupported_reason()
            if reason is not None:
                skipped.append((os.path.basename(path), reason))
                continue
            source_file = os.path.relpath(path, root).replace(os.sep, "/")
            items.append(self.Item(path, index, source_file, sensor_key(path), None, remote_path))

        if not items:
            if skipped:
                raise RuntimeError(
                    f"Found {len(skipped)} Seyond lidar pcap capture(s), but none can be imported: "
                    f"{skipped[0][1]}."
                )
            for name, reason in not_seyond:
                logger.warning(f"'{name}' is not a Seyond lidar capture ({reason}), so it is not imported.")
            return False

        for name, reason in skipped:
            logger.warning(f"Skipped Seyond capture '{name}': {reason}.")
        for name, reason in not_seyond:
            logger.warning(f"'{name}' is not a Seyond lidar capture ({reason}), so it is not imported.")
        if other_files:
            shown = ", ".join(sorted(os.path.basename(p) for p in other_files)[:10])
            more = f" and {len(other_files) - 10} more" if len(other_files) > 10 else ""
            logger.warning(
                f"Importing the Seyond captures only: {len(other_files)} other file(s) are not "
                f"imported ({shown}{more}). Import other point clouds separately."
            )

        self._config = SensorsConfig.load(config_path)
        unmounted = set()
        for item in items:
            item.matrix = self._config.matrix_for(item.sensor, get_file_name(item.path))
            if item.matrix is None:
                unmounted.add(item.sensor)
        if unmounted:
            logger.warning(
                f"No sensor mount for {sorted(unmounted)}: their points stay in the sensor's own axes. "
                f"Give each sensor's extrinsics in '{SENSORS_CONFIG_NAME}' to import them in world axes; "
                "a mount cannot be changed once the frames have labels."
            )

        items.sort(key=lambda item: (item.sensor, item.start_time, item.source_file))
        self._items = items
        if labels_dir:
            self._load_labels(labels_dir)
        logger.info(
            f"Found {len(items)} Seyond lidar capture(s) with {self.frames_count} frames "
            f"from {len(set(i.sensor for i in items))} sensor(s)."
        )
        return True

    def _scan_input(self):
        pcap_paths, config_path, labels_dir, has_sly_project = [], None, None, False
        other_files = []
        for root, dirs, files in os.walk(self._input_data):
            dirs.sort()
            if LABELS_DIR_NAME in os.path.relpath(root, self._input_data).split(os.sep):
                if os.path.basename(root) == LABELS_DIR_NAME and LABELS_META_NAME in files:
                    labels_dir = labels_dir or root
                continue
            for file in sorted(files):
                path = os.path.join(root, file)
                if get_file_ext(file).lower() == PCAP_EXT:
                    pcap_paths.append(path)
                elif file == SENSORS_CONFIG_NAME and config_path is None:
                    config_path = path
                elif file == "meta.json" and self._is_project_meta(path):
                    has_sly_project = True
                elif file not in JUNK_FILES:
                    other_files.append(path)
        return pcap_paths, config_path, labels_dir, has_sly_project, other_files

    @staticmethod
    def _is_project_meta(path: str) -> bool:
        """A Supervisely project's meta.json, not any file that happens to have the name."""
        if os.path.getsize(path) == 0:
            return True  # a linked stand-in: cannot be read here, assume the project
        try:
            data = load_json_file(path)
        except Exception:
            return False
        return isinstance(data, dict) and ("classes" in data or "tags" in data)

    def _remote_path(self, path: str) -> Optional[str]:
        if not self.upload_as_links or not self.remote_files_map:
            return None
        return self.remote_files_map.get(os.path.abspath(path))

    def _index_capture(self, path: str, remote_path: Optional[str]) -> SeyondPcapIndex:
        """Index a capture; a linked one is streamed to a temporary file and removed after.

        :raises SeyondPcapError: If the file is not a classic pcap file.
        """
        local_path, temp_path = path, None
        try:
            if remote_path is not None and os.path.getsize(path) == 0:
                # the local file is an empty stand-in for the one in the bucket
                fd, temp_path = tempfile.mkstemp(suffix=PCAP_EXT)
                os.close(fd)
                logger.info(f"Reading '{remote_path}' to index its frames...")
                self._api.storage.download(self._team_id, remote_path, temp_path)
                local_path = temp_path
            return index_pcap(local_path)
        finally:
            if temp_path is not None:
                silent_remove(temp_path)

    def _load_labels(self, labels_dir: str) -> None:
        meta_json = load_json_file(os.path.join(labels_dir, LABELS_META_NAME))
        self._meta = ProjectMeta.from_json(meta_json)
        for root, _, files in os.walk(labels_dir):
            if LABELS_FILE_NAME not in files:
                continue
            labels = load_json_file(os.path.join(root, LABELS_FILE_NAME))
            frames = {f["frame"]: (f["file"], f["frameInFile"]) for f in labels.get("frames", [])}
            if labels.get("episodeAnnotation") is not None:
                self._episode_labels.append((labels["episodeAnnotation"], frames))
            by_name = {f["name"]: (f["file"], f["frameInFile"]) for f in labels.get("frames", [])}
            for name, ann_json in (labels.get("annotations") or {}).items():
                if name in by_name:
                    self._single_labels[by_name[name]] = ann_json
        logger.info(f"Found labels exported with the captures in '{labels_dir}'.")

    # ---- upload ----

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        """Upload each capture once and add one item per frame that reuses it."""
        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)
        groups = self._group_items()
        dataset_info = api.dataset.get_info_by_id(dataset_id)
        used_names = set()
        if len(groups) > 1:
            nested = api.dataset.get_list(dataset_info.project_id, parent_id=dataset_id)
            used_names = {ds.name for ds in nested}

        if log_progress:
            progress, progress_cb = self.get_progress(self.frames_count, "Adding lidar frames...")
        else:
            progress, progress_cb = None, None

        for name, items in groups:
            # each dataset is created just before its upload, and a group that fails is taken
            # out again: what an interrupted import leaves is whole episodes and captures only
            dataset = self._target_dataset(api, dataset_info, name, used_names, len(groups) == 1)
            added_ids: List[int] = []
            try:
                self._upload_group(
                    api, dataset, items, meta, renamed_classes, renamed_tags, progress_cb, added_ids
                )
            except Exception:
                self._remove_partial_group(api, dataset, dataset_info, added_ids)
                raise
            logger.info(f"Dataset '{dataset.name}' (ID: {dataset.id}) has been uploaded.")

        if progress is not None and is_development():
            progress.close()

    @staticmethod
    def _remove_partial_group(api: Api, dataset, dataset_info, added_ids: List[int]) -> None:
        try:
            if dataset.id != dataset_info.id:
                api.dataset.remove(dataset.id)
            elif added_ids:
                api.pointcloud.remove_batch(added_ids)
        except Exception as e:
            logger.warning(f"Could not remove the partly imported dataset '{dataset.name}': {e}")

    def _upload_group(
        self, api, dataset, items, meta, renamed_classes, renamed_tags, progress_cb, added_ids
    ) -> None:
        """Add the captures of one dataset (an episode, or a capture) and their labels."""
        existing_names = {info.name for info in api.pointcloud.get_list(dataset.id)}
        frame_to_pointcloud_ids: Dict[int, int] = {}
        frame_keys: Dict[Tuple[str, int], int] = {}
        frame_offset = 0
        for item in items:
            infos = self._add_capture(api, dataset.id, item, frame_offset, existing_names)
            added_ids.extend(info.id for info in infos)
            for frame, info in zip(item.index.frames, infos):
                item_frame = frame_offset + frame.index if self._is_pcd_episode else frame.index
                frame_to_pointcloud_ids[item_frame] = info.id
                frame_keys[(item.source_file, frame.index)] = item_frame
                if not self._is_pcd_episode:
                    self._append_single_labels(
                        api, info.id, (item.source_file, frame.index), meta,
                        renamed_classes, renamed_tags,
                    )
            if self._is_pcd_episode:
                frame_offset += item.frame_count
            if progress_cb is not None:
                progress_cb(item.frame_count)

        if self._is_pcd_episode:
            ann = self._episode_annotation(
                frame_offset, frame_keys, meta, renamed_classes, renamed_tags
            )
            api.pointcloud_episode.annotation.append(
                dataset.id, ann, frame_to_pointcloud_ids, KeyIdMap()
            )

    def _group_items(self) -> List[Tuple[str, List[Item]]]:
        """Point clouds: one dataset per capture."""
        return [(get_file_name(item.path), [item]) for item in self._items]

    @staticmethod
    def _target_dataset(api: Api, dataset_info, name: str, used_names: set, only_one: bool):
        """The target dataset for one group, or a new dataset nested in it when there are several."""
        if only_one:
            return dataset_info
        ds_name = generate_free_name(used_names, name, extend_used_names=True)
        return api.dataset.create(
            dataset_info.project_id, ds_name, change_name_if_conflict=True, parent_id=dataset_info.id
        )

    def _capture_data(self, api: Api, item: Item) -> Tuple[str, str]:
        """How the items reference the capture: its link in cloud storage, or its uploaded hash.

        A capture from Team Files is uploaded by hash too, not added by ``teamFileId``: the dataset
        counts a Team Files entry's size once per item, so 600 frames of one file would count it
        600 times, while one request of hash entries counts the file once.
        """
        if item.remote_path is not None and item.remote_path.startswith(_CLOUD_STORAGE_PREFIXES):
            return ApiField.LINK, item.remote_path

        path, temp_path = item.path, None
        try:
            if item.remote_path is not None and os.path.getsize(item.path) == 0:
                # a Team Files path given as a "link": its local file is an empty stand-in
                fd, temp_path = tempfile.mkstemp(suffix=PCAP_EXT)
                os.close(fd)
                api.storage.download(self._team_id, item.remote_path, temp_path)
                path = temp_path
            file_hash = api.pointcloud._upload_data_bulk(
                lambda p: open(p, "rb"), get_file_hash, [path], None
            )[0]
        finally:
            if temp_path is not None:
                silent_remove(temp_path)
        return ApiField.HASH, file_hash

    def _add_capture(
        self, api: Api, dataset_id: int, item: Item, frame_offset: int, existing_names: set
    ):
        """Add every frame of a capture in one request, so the dataset counts the file once."""
        key, value = self._capture_data(api, item)
        width = max(4, len(str(item.frame_count - 1)))
        stem = get_file_name(item.path)
        entries = []
        for frame in item.index.frames:
            meta = {
                "frame": frame_offset + frame.index if self._is_pcd_episode else frame.index,
                "timestamp": frame.timestamp,
                "source": {
                    "file": item.source_file,
                    "frameInFile": frame.index,
                    "sensor": item.sensor,
                },
            }
            if item.matrix is not None:
                meta["sensorsData"] = {"extrinsicMatrix": item.matrix}
            name = generate_free_name(
                existing_names,
                f"{stem}_{frame.index:0{width}d}{PCAP_EXT}",
                with_ext=True,
                extend_used_names=True,
            )
            entries.append({ApiField.NAME: name, key: value, ApiField.META: meta})

        response = api.post(
            "point-clouds.bulk.add",
            {ApiField.DATASET_ID: dataset_id, ApiField.POINTCLOUDS: entries},
        )
        infos = [api.pointcloud._convert_json_info(info) for info in response.json()]
        by_name = {info.name: info for info in infos}
        return [by_name[entry[ApiField.NAME]] for entry in entries]

    # ---- labels from a previous export ----

    def _append_single_labels(self, api, pointcloud_id, key, meta, renamed_classes, renamed_tags):
        ann_json = self._single_labels.get(key)
        if ann_json is None:
            return
        ann_json = rename_in_json(copy.deepcopy(ann_json), renamed_classes or {}, renamed_tags or {})
        ann = PointcloudAnnotation.from_json(ann_json, meta)
        api.pointcloud.annotation.append(pointcloud_id, ann, KeyIdMap())

    def _episode_annotation(
        self,
        frames_count: int,
        frame_keys: Dict[Tuple[str, int], int],
        meta: ProjectMeta,
        renamed_classes: dict,
        renamed_tags: dict,
    ) -> PointcloudEpisodeAnnotation:
        """The episode's labels: exported figures moved to the frames they land on now."""
        objects, frames, tags = {}, {}, []
        for ann_json, exported_frames in self._episode_labels:
            remap = {
                old: frame_keys[key] for old, key in exported_frames.items() if key in frame_keys
            }
            if not remap:
                continue
            used_objects = set()
            for frame in ann_json.get("frames", []):
                new_index = remap.get(frame["index"])
                if new_index is None:
                    continue
                for figure in frame.get("figures", []):
                    frames.setdefault(new_index, []).append(copy.deepcopy(figure))
                    used_objects.add(figure.get("objectKey"))
            for obj in ann_json.get("objects", []):
                if obj.get("key") in used_objects:
                    obj = copy.deepcopy(obj)
                    obj["tags"] = self._remap_tags(obj.get("tags", []), remap)
                    objects[obj["key"]] = obj
            tags.extend(self._remap_tags(ann_json.get("tags", []), remap))

        ann_json = {
            "framesCount": frames_count,
            "description": "",
            "key": uuid.uuid4().hex,
            "tags": tags,
            "objects": list(objects.values()),
            "frames": [{"index": i, "figures": figs} for i, figs in sorted(frames.items())],
        }
        ann_json = rename_in_json(ann_json, renamed_classes or {}, renamed_tags or {})
        return PointcloudEpisodeAnnotation.from_json(ann_json, meta)

    @staticmethod
    def _remap_tags(tags: List[dict], remap: Dict[int, int]) -> List[dict]:
        result = []
        for tag in tags:
            tag = copy.deepcopy(tag)
            frame_range = tag.get("frameRange")
            if frame_range is not None:
                start, end = remap.get(frame_range[0]), remap.get(frame_range[1])
                if start is None or end is None:
                    logger.warning(
                        f"Tag '{tag.get('name')}' spans frames that are not in this episode, skipped."
                    )
                    continue
                tag["frameRange"] = [start, end]
            result.append(tag)
        return result
