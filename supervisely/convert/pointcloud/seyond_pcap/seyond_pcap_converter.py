import copy
import os
import tempfile
import uuid
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from supervisely import (
    Api,
    PointcloudAnnotation,
    PointcloudEpisodeAnnotation,
    ProjectMeta,
    generate_free_name,
    is_development,
    logger,
)
from supervisely._utils import get_bytes_hash
from supervisely.api.module_api import ApiField
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.convert.pointcloud.seyond_pcap.seyond_pcap_helper import (
    ANGLE_TABLE_KIND,
    ATTACHMENT_ENTITY_TYPE,
    LABELS_DIR_NAME,
    LABELS_FILE_NAME,
    LABELS_META_NAME,
    LOG_EXT,
    SENSORS_CONFIG_NAME,
    SensorsConfig,
    _AngleTable,
    parse_capture_name,
    parse_log_name,
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
    _ANGLE_TABLE_MAX_PACKET,
    ANGLE_TABLE_EXT,
    PCAP_EXT,
    SeyondPcapError,
    SeyondPcapIndex,
    _find_angle_tables,
    _is_angle_table,
    index_pcap,
)
from supervisely.project.project_settings import LabelingInterface
from supervisely.project.project_type import ProjectType
from supervisely.video_annotation.key_id_map import KeyIdMap

_CLOUD_STORAGE_PREFIXES = ("s3://", "azure://", "google://", "agent://", "fs://")
# where an angle table waits in Team Files until it is attached to a dataset
_TABLE_UPLOAD_DIR = "/tmp/supervisely/seyond-angle-tables/"


class _TableSource(NamedTuple):
    """A file in the input that holds angle tables: a vehicle log, or one table saved as a file."""

    path: str
    folder: str
    #: (device, timestamp) of a log's recording; None for a table file
    recording: Optional[Tuple[str, str]]
    tables: Dict[str, _AngleTable]


class SeyondPcapConverter(PointcloudConverter):
    """Imports Seyond (Innovusion) lidar captures: raw libpcap files of the sensor's UDP output.

    A capture is stored once, by upload, as a link to cloud storage, or by its Team Files id, and
    every lidar frame in it becomes an item that reuses it. The labeling tool decodes each item's
    frame from the file, so nothing is converted and nothing is copied per frame.

    - Each item's ``meta`` holds ``frame``, ``timestamp`` (the capture time of the frame's first
      packet, UTC), ``sensorsData.extrinsicMatrix`` when ``sensors.json`` gives the sensor's mount,
      and ``source``: ``{"file", "frameInFile", "sensor"}``. Current instances drop ``source``;
      the export works the capture and frame out from ``frame`` and the item name without it.
    - Point clouds: one dataset per capture, ``meta.frame`` is the frame in the file.
    - Point cloud episodes (:class:`SeyondPcapEpisodesConverter`): the captures of each sensor are
      chained by time into episodes, ``meta.frame`` continues across them.
    - Labels written next to the captures by the export (``supervisely_labels/``) are re-attached.
    - A Robin W capture is decoded with its sensor's angle table, which the capture does not hold.
      The table is read from the vehicle log of the recording (``{device}--{timestamp}.log`` next
      to the captures) or from ``<serial>.anglehv`` (what the export writes), and is kept with
      each dataset that has the sensor's frames, as an attachment (see :meth:`_tables_from_input`).
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
            #: The sensor's angle table, for a Robin W capture.
            self.table: Optional[_AngleTable] = None

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
        pcap_paths, config_path, labels_dir, has_sly_project, other_files, table_paths = (
            self._scan_input()
        )
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
            if reason is None and index.needs_angle_table and index.serial is None:
                reason = (
                    "a Robin W capture without status packets: the serial number that picks "
                    "the sensor's angle table is unknown"
                )
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

        if any(item.index.needs_angle_table for item in items):
            self._tables_from_input(items, table_paths)

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
        other_files, table_paths = [], []
        for root, dirs, files in os.walk(self._input_data):
            dirs.sort()
            if LABELS_DIR_NAME in os.path.relpath(root, self._input_data).split(os.sep):
                if os.path.basename(root) == LABELS_DIR_NAME and LABELS_META_NAME in files:
                    labels_dir = labels_dir or root
                continue
            for file in sorted(files):
                path = os.path.join(root, file)
                ext = get_file_ext(file).lower()
                if ext == PCAP_EXT:
                    pcap_paths.append(path)
                elif ext in (LOG_EXT, ANGLE_TABLE_EXT) and not file.startswith("._"):
                    # where Robin W angle tables come from, never items
                    table_paths.append(path)
                elif file == SENSORS_CONFIG_NAME and config_path is None:
                    config_path = path
                elif file == "meta.json" and self._is_project_meta(path, self.upload_as_links):
                    has_sly_project = True
                elif (
                    file != SENSORS_CONFIG_NAME
                    and file not in JUNK_FILES
                    and not file.startswith("._")
                    and not any(part in JUNK_FILES for part in os.path.relpath(root, self._input_data).split(os.sep))
                ):
                    other_files.append(path)
        return pcap_paths, config_path, labels_dir, has_sly_project, other_files, table_paths

    @staticmethod
    def _is_project_meta(path: str, linked: bool) -> bool:
        """A Supervisely project's meta.json, not any file that happens to have the name."""
        if os.path.getsize(path) == 0:
            return linked  # a linked stand-in cannot be read here: assume the project
        try:
            # the same test the Supervisely format converters apply to their key file
            ProjectMeta.from_json(load_json_file(path))
        except Exception:
            return False
        return True

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

    def _local_copy(self, path: str, suffix: str) -> Tuple[str, Optional[str]]:
        """A local path to read the file from, and the temporary file to remove after, if any.

        A file given as a link is an empty stand-in for the one in the bucket or Team Files.
        """
        remote_path = self._remote_path(path)
        if remote_path is None or os.path.getsize(path) != 0:
            return path, None
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            self._api.storage.download(self._team_id, remote_path, temp_path)
        except Exception:
            silent_remove(temp_path)
            raise
        return temp_path, temp_path

    # ---- Robin W angle tables ----

    def _tables_from_input(self, items: List[Item], table_paths: List[str]) -> None:
        """Give each Robin W capture its sensor's angle table from the input, where it has one.

        The table of a sensor can change (a firmware update, a recalibration), so a capture takes
        the one closest to its own recording, matched by the serial number in its status packets:

        1. the log of the same recording: ``{device}--{timestamp}.log`` in the capture's folder;
        2. ``<serial>.anglehv`` in the capture's folder;
        3. any other log or table file in the input: the latest recording that started no later
           than the capture, else the first one by path.

        A capture left without one gets the latest table the project, then the team, already
        keeps for that sensor when it is uploaded (:meth:`_tables_from_attachments`).
        """
        serials = {item.index.serial for item in items if item.index.needs_angle_table}
        sources: List[_TableSource] = []
        for path in sorted(table_paths):
            tables = self._read_tables(path, serials)
            if not tables:
                continue
            parsed = parse_log_name(path) if get_file_ext(path).lower() == LOG_EXT else None
            recording = (parsed["device"], parsed["timestamp"]) if parsed else None
            sources.append(_TableSource(path, os.path.dirname(path), recording, tables))

        used = set()
        for item in items:
            serial = item.index.serial
            if not item.index.needs_angle_table:
                continue
            candidates = [s for s in sources if serial in s.tables]
            if not candidates:
                continue
            parsed = parse_capture_name(item.path)
            recording = (parsed["device"], parsed["timestamp"]) if parsed else None
            folder = os.path.dirname(item.path)

            own_log = [
                s for s in candidates
                if s.folder == folder and recording is not None and s.recording == recording
            ]
            own_file = [
                s for s in candidates
                if s.folder == folder and s.recording is None and get_file_name(s.path) == serial
            ]
            earlier = [
                s for s in candidates
                if recording is not None and s.recording is not None and s.recording[1] <= recording[1]
            ]
            if own_log or own_file:
                source = (own_log or own_file)[0]
            elif earlier:
                source = max(earlier, key=lambda s: s.recording[1])
            else:
                source = candidates[0]
            item.table = source.tables[serial]
            if (serial, source.path) not in used:
                used.add((serial, source.path))
                logger.info(
                    f"The angle table of Robin W sensor {serial} comes from "
                    f"'{os.path.relpath(source.path, self._input_data)}'."
                )

    def _read_tables(self, path: str, serials: set) -> Dict[str, _AngleTable]:
        """The angle tables a log or a table file holds for the given serial numbers."""
        name = os.path.relpath(path, self._input_data)
        is_table_file = get_file_ext(path).lower() == ANGLE_TABLE_EXT
        if is_table_file and get_file_name(path) not in serials:
            return {}
        local_path, temp_path = self._local_copy(path, get_file_ext(path))
        try:
            if is_table_file:
                with open(local_path, "rb") as f:
                    data = f.read(_ANGLE_TABLE_MAX_PACKET + 1)
                if not _is_angle_table(data):
                    logger.warning(f"'{name}' is not a Robin W angle table, so it is not used.")
                    return {}
                found = [(get_file_name(path), data)]
            else:
                logger.info(f"Reading Robin W angle tables from '{name}'...")
                found = _find_angle_tables(local_path)
        finally:
            if temp_path is not None:
                silent_remove(temp_path)

        tables: Dict[str, _AngleTable] = {}
        changed = set()
        for serial, data in found:
            if serial not in serials:
                continue
            table = tables.get(serial)
            if table is None:
                tables[serial] = _AngleTable(serial, get_bytes_hash(data), data, name)
            elif table.hash != get_bytes_hash(data) and serial not in changed:
                changed.add(serial)
                logger.warning(
                    f"'{name}' has more than one angle table for Robin W sensor {serial}: "
                    "the first one is used."
                )
        return tables

    def _tables_from_attachments(self, api: Api, project_id: int) -> None:
        """Give the Robin W captures still without a table the latest one kept for their sensor.

        The project's datasets are looked at first, then those of the team's other point cloud
        projects. A capture with no table anywhere is not imported: the tool could not show it.
        """
        missing = {
            item.index.serial
            for item in self._items
            if item.index.needs_angle_table and item.table is None
        }
        if missing:
            found = self._latest_tables(api, [project_id], missing)
            if missing - set(found):
                found.update(self._latest_tables(api, self._team_projects(api, project_id), missing - set(found)))
            for item in self._items:
                if item.index.needs_angle_table and item.table is None:
                    item.table = found.get(item.index.serial)
            for serial, table in sorted(found.items()):
                logger.info(
                    f"The vehicle log has no angle table for Robin W sensor {serial}: using the "
                    f"latest one kept for it ({table.source})."
                )

        kept = []
        for item in self._items:
            if item.index.needs_angle_table and item.table is None:
                logger.warning(
                    f"Skipped Seyond capture '{os.path.basename(item.path)}': no angle table for "
                    f"Robin W sensor {item.index.serial}. Upload the vehicle log ({LOG_EXT}) of "
                    "the recording with its captures."
                )
                continue
            kept.append(item)
        if not kept:
            raise RuntimeError(
                "None of the Seyond captures can be imported: the Robin W sensors' angle tables "
                f"are missing. Upload the vehicle log ({LOG_EXT}) of each recording with its captures."
            )
        self._items = kept

    @staticmethod
    def _list_tables(api: Api, **where) -> List[dict]:
        """The angle tables kept with a dataset or a project (``datasetId`` or ``projectId``)."""
        entities = api.pointcloud.get_list_all_pages(
            "entities.list",
            {**where, ApiField.TYPE: ATTACHMENT_ENTITY_TYPE},
            convert_json_info_cb=lambda entity: entity,
        )
        return [
            e
            for e in entities
            if isinstance(e.get(ApiField.META), dict)
            and e[ApiField.META].get("kind") == ANGLE_TABLE_KIND
            and e[ApiField.META].get("serial")
        ]

    def _latest_tables(self, api: Api, project_ids: List[int], serials: set) -> Dict[str, _AngleTable]:
        latest: Dict[str, dict] = {}
        for project_id in project_ids:
            for entity in self._list_tables(api, projectId=project_id):
                serial = entity[ApiField.META]["serial"]
                if serial not in serials or not entity.get(ApiField.HASH):
                    continue
                if serial not in latest or entity[ApiField.CREATED_AT] > latest[serial][ApiField.CREATED_AT]:
                    latest[serial] = entity
        return {
            serial: _AngleTable(
                serial,
                entity[ApiField.HASH],
                None,
                f"dataset {entity[ApiField.DATASET_ID]} of project {entity[ApiField.PROJECT_ID]}",
            )
            for serial, entity in latest.items()
        }

    @staticmethod
    def _team_projects(api: Api, project_id: int) -> List[int]:
        """The team's other point cloud and point cloud episodes projects."""
        project = api.project.get_info_by_id(project_id)
        types = (str(ProjectType.POINT_CLOUDS), str(ProjectType.POINT_CLOUD_EPISODES))
        result = []
        for workspace in api.workspace.get_list(project.team_id):
            for info in api.project.get_list(workspace.id):
                if info.id != project_id and info.type in types:
                    result.append(info.id)
        return result

    def _add_tables(self, api: Api, dataset_id: int, items: List[Item]) -> None:
        """Keep the angle tables of a dataset's Robin W captures with it: one per sensor.

        A table the platform already stores is added by its hash; a new one goes through Team
        Files, which the platform requires for a first upload, and is removed from there after.
        """
        tables = {}
        for item in items:
            if item.table is not None:
                tables[item.table.serial] = item.table
        if not tables:
            return
        kept = {e[ApiField.META]["serial"]: e for e in self._list_tables(api, datasetId=dataset_id)}
        tables = {serial: t for serial, t in tables.items() if serial not in kept}
        if not tables:
            return
        new = [t.hash for t in tables.values() if t.data is not None]
        stored = set(api.pointcloud.check_existing_hashes(new)) if new else set()
        team_id = self._dataset_team_id(api, dataset_id) if len(stored) < len(new) else None

        entities, uploaded = [], []
        try:
            for serial, table in sorted(tables.items()):
                entity = {
                    ApiField.TITLE: table.title,
                    ApiField.TYPE: ATTACHMENT_ENTITY_TYPE,
                    ApiField.META: {"kind": ANGLE_TABLE_KIND, "serial": serial},
                }
                if table.data is None or table.hash in stored:
                    entity[ApiField.HASH] = table.hash
                else:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        local_path = os.path.join(temp_dir, f"{serial}{ANGLE_TABLE_EXT}")
                        with open(local_path, "wb") as f:
                            f.write(table.data)
                        remote_path = f"{_TABLE_UPLOAD_DIR}{uuid.uuid4().hex}/{serial}{ANGLE_TABLE_EXT}"
                        file_info = api.file.upload(team_id, local_path, remote_path)
                    uploaded.append((file_info.team_id, file_info.path))
                    entity[ApiField.TEAM_FILE_ID] = file_info.id
                entities.append(entity)
            api.post(
                "entities.bulk.add",
                {ApiField.DATASET_ID: dataset_id, ApiField.ENTITIES: entities},
            )
        finally:
            for team_id, path in uploaded:
                try:
                    api.file.remove(team_id, path)
                except Exception as e:
                    logger.warning(f"Could not remove the angle table '{path}' from Team Files: {e}")
        for table in tables.values():
            table.data = None  # stored now: the next dataset adds it by hash

    @staticmethod
    def _dataset_team_id(api: Api, dataset_id: int) -> int:
        dataset = api.dataset.get_info_by_id(dataset_id)
        return api.project.get_info_by_id(dataset.project_id).team_id

    def _other_tables_in(self, api: Api, dataset_id: int, items: List[Item]) -> bool:
        """The dataset already keeps a different table for a sensor of these captures."""
        tables = {item.table.serial: item.table.hash for item in items if item.table is not None}
        if not tables:
            return False
        for entity in self._list_tables(api, datasetId=dataset_id):
            serial = entity[ApiField.META]["serial"]
            if serial in tables and entity.get(ApiField.HASH) != tables[serial]:
                return True
        return False

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
        dataset_info = api.dataset.get_info_by_id(dataset_id)
        self._tables_from_attachments(api, dataset_info.project_id)
        groups = self._group_items()
        # a dataset keeps one table per sensor: a target that keeps another one gets a nested dataset
        only_one = len(groups) == 1 and not self._other_tables_in(api, dataset_id, groups[0][1])
        used_names = set()
        if not only_one:
            nested = api.dataset.get_list(dataset_info.project_id, parent_id=dataset_id)
            used_names = {ds.name for ds in nested}

        if log_progress:
            progress, progress_cb = self.get_progress(self.frames_count, "Adding lidar frames...")
        else:
            progress, progress_cb = None, None

        for name, items in groups:
            # each dataset is created just before its upload, and a group that fails is taken
            # out again: what an interrupted import leaves is whole episodes and captures only
            dataset = self._target_dataset(api, dataset_info, name, used_names, only_one)
            added_ids: List[int] = []
            try:
                # the tables first: the tool never finds a Robin W frame without its table. A
                # table stays if the frames fail: removing it would count it off the dataset's
                # items, which it never was, and a new import into the dataset reuses it
                self._add_tables(api, dataset.id, items)
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
                # the point cloud API has no bulk removal; images.bulk.remove takes any entity
                api.image.remove_batch(added_ids)
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
            infos = self._add_capture(api, dataset.id, item, frame_offset, existing_names, added_ids)
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
        self,
        api: Api,
        dataset_id: int,
        item: Item,
        frame_offset: int,
        existing_names: set,
        added_ids: List[int],
    ):
        """Add every frame of a capture in one request, so the dataset counts the file once.

        The ids of the items added are appended to ``added_ids`` as soon as the request returns.
        """
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
        added = response.json()
        added_ids.extend(info[ApiField.ID] for info in added if ApiField.ID in info)
        infos = [api.pointcloud._convert_json_info(info) for info in added]
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
