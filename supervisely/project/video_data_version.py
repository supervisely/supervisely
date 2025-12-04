from __future__ import annotations

from typing import List, Optional, Tuple, Union

import io
import json
import os
import tarfile
import tempfile
import zstd

from supervisely._utils import logger, batched
from supervisely.api.module_api import ApiField
from supervisely.api.project_api import ProjectInfo, ProjectType
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.video.video_api import VideoInfo
from supervisely.io.fs import mkdir, clean_dir, remove_dir
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project import Dataset
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.data_version import DataVersion
from supervisely.task.progress import tqdm_sly
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_annotation import VideoAnnotation


class VideoDataVersion(DataVersion):
    def __init__(self, api):
        """
        Class for managing versions of video projects.
        """
        super().__init__(api)
        self.__version_format = "v2.0.0"

    @staticmethod
    def build_snapshot(
        api,
        project_id: int,
        dataset_ids: Optional[List[int]] = None,
        batch_size: int = 50,
        log_progress: bool = True,
        progress_cb=None,
    ) -> io.BytesIO:
        """
        Wrapper for building a video project snapshot.
        """
        return _build_snapshot(
            api,
            project_id=project_id,
            dataset_ids=dataset_ids,
            batch_size=batch_size,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    @staticmethod
    def restore_snapshot(
        api,
        snapshot_bytes: bytes,
        workspace_id: int,
        project_name: Optional[str] = None,
        with_custom_data: bool = True,
        log_progress: bool = True,
        progress_cb=None,
        skip_missed: bool = False,
    ) -> "ProjectInfo":
        """
        Wrapper for restoring a video project from snapshot.
        """
        return _restore_snapshot(
            api,
            snapshot_bytes=snapshot_bytes,
            workspace_id=workspace_id,
            project_name=project_name,
            with_custom_data=with_custom_data,
            log_progress=log_progress,
            progress_cb=progress_cb,
            skip_missed=skip_missed,
        )

    def create(
        self,
        project_info: Union[ProjectInfo, int],
        version_title: Optional[str] = None,
        version_description: Optional[str] = None,
    ) -> Optional[int]:
        """
        Create a new project version.
        Returns the ID of the new version.
        If the project is already on the latest version, returns the latest version ID.
        If the project version cannot be created, returns None.

        :param project_info: ProjectInfo object or project ID
        :type project_info: Union[ProjectInfo, int]
        :param version_title: Version title
        :type version_title: Optional[str]
        :param version_description: Version description
        :type version_description: Optional[str]
        :return: Version ID
        :rtype: int
        """
        if isinstance(project_info, int):
            project_info = self._api.project.get_info_by_id(project_info)

        if project_info.type != ProjectType.VIDEOS.value:
            raise ValueError(f"Project with id {project_info.id} is not a video project")

        if (
            "app.supervise.ly" in self._api.server_address
            or "app.supervisely.com" in self._api.server_address
        ):
            if self._api.team.get_info_by_id(project_info.team_id).usage.plan == "free":
                logger.warning(
                    "Project versioning is not available for teams with Free plan. "
                    "Please upgrade to Pro to enable versioning."
                )
                return None

        self.initialize(project_info)
        path = self._generate_save_path()
        latest = self._get_latest_id()
        try:
            version_id, commit_token = self.reserve(project_info.id)
            # @TODO: remove log
            logger.debug(f"version_id: {version_id}, commit_token: {commit_token}")
        except Exception as e:
            logger.error(f"Failed to reserve video version. Exception: {e}")
            return None
        if version_id is None and commit_token is None:
            return latest
        try:
            file_info = self._compress_and_upload(path)
            latest_number = (
                int(self.versions[str(latest)]["number"])
                if (latest and str(latest) in self.versions)
                else 0
            )
            self.versions[version_id] = {
                "path": path,
                "updated_at": project_info.updated_at,
                "previous": latest,
                "number": latest_number + 1,
                "schema": self.__version_format,
            }
            self.versions["latest"] = version_id
            self.set_map(project_info, initialize=False)
            self.commit(
                version_id,
                commit_token,
                project_info.updated_at,
                file_info.id,
                title=version_title,
                description=version_description,
            )
            return version_id

        except Exception as e:
            if self.cancel_reservation(version_id, commit_token):
                logger.error(
                    f"Video version creation failed. Reservation was cancelled. Exception: {e}"
                )
            else:
                logger.error(
                    "Failed to cancel video version reservation when handling exception. "
                    "You can cancel your reservation on the web under the Versions tab of the project. "
                    f"Exception: {e}"
                )
            return None

    def restore(
        self,
        project_info: Union[ProjectInfo, int],
        version_id: Optional[int] = None,
        version_num: Optional[int] = None,
        skip_missed_entities: bool = False,
    ) -> Optional[ProjectInfo]:
        """
        Restore project to a specific version.
        Version can be specified by ID or number.

        :param project_info: ProjectInfo object or project ID
        :type project_info: Union[ProjectInfo, int]
        :param version_id: Version ID
        :type version_id: Optional[int]
        :param version_num: Version number
        :type version_num: Optional[int]
        :param skip_missed_entities: Skip missed Images
        :type skip_missed_entities: bool, default False
        :return: ProjectInfo object of the restored project
        :rtype: ProjectInfo or None
        """

        if version_id is None and version_num is None:
            raise ValueError("Either version_id or version_num must be provided")

        if isinstance(project_info, int):
            project_info = self._api.project.get_info_by_id(project_info)

        if project_info.type != ProjectType.VIDEOS.value:
            raise ValueError(f"Project with id {project_info.id} is not a video project")

        self.initialize(project_info)

        if version_num is not None:
            resolved_id = None
            for key, value in self.versions.items():
                if not isinstance(value, dict):
                    continue
                if value.get("number") == version_num:
                    resolved_id = key
                    break
            if resolved_id is None:
                raise ValueError(f"Version {version_num} does not exist for this video project")
            version_id = int(resolved_id)
        else:
            if str(version_id) not in self.versions:
                raise ValueError(f"Version {version_id} does not exist for this video project")

        vinfo = self.versions[str(version_id)]
        backup_path = vinfo.get("path")
        schema = vinfo.get("schema", self.__version_format)

        if schema != self.__version_format:
            raise RuntimeError(
                f"Unsupported video version schema '{schema}' "
                f"(expected '{self.__version_format}')"
            )

        if backup_path is None:
            logger.warning(
                f"Video project can't be restored to version {vinfo.get('number')} "
                f"because it doesn't have restore point."
            )
            return None

        snapshot_io = self._download_snapshot(backup_path)
        snapshot_bytes = snapshot_io.getvalue()
        new_project_info = self.restore_snapshot(
            self._api,
            snapshot_bytes,
            workspace_id=self.project_info.workspace_id,
            project_name=None,
            with_custom_data=True,
            log_progress=False,
            progress_cb=None,
            skip_missed=skip_missed_entities,
        )
        return new_project_info

    def _compress_and_upload(self, path: str) -> dict:
        """
        Save video project snapshot in archive to the Team Files.

        :param path: Destination path in Team Files
        :return: File info
        :rtype: dict
        """
        temp_dir = tempfile.mkdtemp()
        try:
            snapshot_io = self.build_snapshot(
                self._api,
                self.project_info.id,
                dataset_ids=None,
                batch_size=50,
                log_progress=False,
                progress_cb=None,
            )

            local_archive = os.path.join(temp_dir, "snapshot.tar.zst")
            with open(local_archive, "wb") as f:
                f.write(snapshot_io.read())

            file_info = self._api.file.upload(
                self.project_info.team_id,
                local_archive,
                path,
            )
            if file_info is None:
                raise RuntimeError("Failed to upload video version snapshot to Team Files")

            return file_info
        finally:
            try:
                remove_dir(temp_dir)
            except Exception:
                pass

    def _download_snapshot(self, path: str) -> io.BytesIO:
        """
        Download stored snapshot (.tar.zst) for a video project version into memory.
        """
        import io
        import os
        import tempfile
        from supervisely.io.fs import remove_dir

        temp_dir = tempfile.mkdtemp()
        local_path = os.path.join(temp_dir, "download.tar.zst")
        try:
            self._api.file.download(self.project_info.team_id, path, local_path)
            with open(local_path, "rb") as f:
                data = f.read()
            return io.BytesIO(data)
        except Exception as e:
            raise RuntimeError(f"Failed to download video version snapshot: {e}")
        finally:
            remove_dir(temp_dir)


def _build_snapshot(
    api,
    project_id: int,
    dataset_ids: Optional[List[int]] = None,
    batch_size: int = 50,
    log_progress: bool = True,
    progress_cb=None,
) -> io.BytesIO:
    """
    Create a video project snapshot in Arrow/Parquet+tar.zst format and return it as BytesIO.
    """
    try:
        import pyarrow
        import pyarrow.parquet as parquet
    except Exception as e:
        raise RuntimeError(
            "pyarrow is required to build video snapshot. Please install pyarrow."
        ) from e

    project_info = api.project.get_info_by_id(project_id)
    meta = ProjectMeta.from_json(api.project.get_meta(project_id, with_settings=True))
    key_id_map = KeyIdMap()

    tmp_root = tempfile.mkdtemp()
    payload_dir = os.path.join(tmp_root, "payload")
    mkdir(payload_dir)

    try:
        # project_info / meta
        proj_info_path = os.path.join(payload_dir, "project_info.json")
        dump_json_file(project_info._asdict(), proj_info_path)

        proj_meta_path = os.path.join(payload_dir, "project_meta.json")
        dump_json_file(meta.to_json(), proj_meta_path)

        datasets_rows: List[dict] = []
        videos_rows: List[dict] = []
        objects_rows: List[dict] = []
        figures_rows: List[dict] = []

        dataset_ids_filter = set(dataset_ids) if dataset_ids is not None else None

        for parents, ds_info in api.dataset.tree(project_id):
            if dataset_ids_filter is not None and ds_info.id not in dataset_ids_filter:
                continue

            full_path = Dataset._get_dataset_path(ds_info.name, parents)
            datasets_rows.append(
                {
                    "src_dataset_id": ds_info.id,
                    "parent_src_dataset_id": ds_info.parent_id,
                    "name": ds_info.name,
                    "full_path": full_path,
                    "description": ds_info.description,
                    "custom_data": (
                        json.dumps(ds_info.custom_data) if ds_info.custom_data else None
                    ),
                }
            )

            videos = api.video.get_list(ds_info.id)
            ds_progress = progress_cb
            if log_progress and progress_cb is None:
                ds_progress = tqdm_sly(
                    desc=f"Collecting videos from '{ds_info.name}'",
                    total=len(videos),
                )

            for batch in batched(videos, batch_size):
                video_ids = [v.id for v in batch]
                ann_jsons = api.video.annotation.download_bulk(ds_info.id, video_ids)

                for video_info, ann_json in zip(batch, ann_jsons):
                    if video_info.name != ann_json[ApiField.VIDEO_NAME]:
                        raise RuntimeError(
                            "Error in api.video.annotation.download_bulk: broken order"
                        )

                    frames_to_timecodes = getattr(video_info, "frames_to_timecodes", None)
                    frames_to_timecodes_json = (
                        json.dumps(frames_to_timecodes) if frames_to_timecodes else None
                    )
                    meta_json = (
                        json.dumps(getattr(video_info, "meta", None))
                        if getattr(video_info, "meta", None)
                        else None
                    )
                    custom_data_json = (
                        json.dumps(video_info.custom_data)
                        if getattr(video_info, "custom_data", None)
                        else None
                    )

                    videos_rows.append(
                        {
                            "src_video_id": video_info.id,
                            "src_dataset_id": ds_info.id,
                            "name": video_info.name,
                            "hash": getattr(video_info, "hash", None),
                            "link": getattr(video_info, "link", None),
                            "frames_count": getattr(video_info, "frames_count", None),
                            "frame_width": getattr(video_info, "frame_width", None),
                            "frame_height": getattr(video_info, "frame_height", None),
                            "frames_to_timecodes": frames_to_timecodes_json,
                            "meta": meta_json,
                            "custom_data": custom_data_json,
                            "created_at": getattr(video_info, "created_at", None),
                            "updated_at": getattr(video_info, "updated_at", None),
                            "ann_json": json.dumps(ann_json),
                        }
                    )

                    video_ann = VideoAnnotation.from_json(ann_json, meta, key_id_map)
                    obj_key_to_src_id: dict[str, int] = {}
                    for obj in video_ann.objects:
                        src_obj_id = len(objects_rows) + 1
                        obj_key_to_src_id[obj.key().hex] = src_obj_id
                        objects_rows.append(
                            {
                                "src_object_id": src_obj_id,
                                "src_video_id": video_info.id,
                                "class_name": obj.obj_class.name,
                                "key": obj.key().hex,
                                "tags_json": (
                                    json.dumps(obj.tags.to_json()) if obj.tags is not None else None
                                ),
                            }
                        )

                    for frame in video_ann.frames:
                        for fig in frame.figures:
                            parent_key = fig.parent_object.key().hex
                            src_obj_id = obj_key_to_src_id.get(parent_key)
                            if src_obj_id is None:
                                logger.warning(
                                    f"Figure parent object with key '{parent_key}' "
                                    f"not found in objects for video '{video_info.name}'"
                                )
                                continue
                            figures_rows.append(
                                {
                                    "src_figure_id": len(figures_rows) + 1,
                                    "src_object_id": src_obj_id,
                                    "src_video_id": video_info.id,
                                    "frame_index": frame.index,
                                    "geometry_type": fig.geometry.geometry_name(),
                                    "geometry_json": json.dumps(fig.geometry.to_json()),
                                }
                            )

                if ds_progress is not None:
                    ds_progress(len(batch))

        # key_id_map.json
        key_id_map_path = os.path.join(payload_dir, "key_id_map.json")
        key_id_map.dump_json(key_id_map_path)

        # Arrow schemas
        tables_meta = []
        datasets_schema = pyarrow.schema(
            [
                ("src_dataset_id", pyarrow.int64()),
                ("parent_src_dataset_id", pyarrow.int64()),
                ("name", pyarrow.utf8()),
                ("full_path", pyarrow.utf8()),
                ("description", pyarrow.utf8()),
                ("custom_data", pyarrow.utf8()),
            ]
        )

        videos_schema = pyarrow.schema(
            [
                ("src_video_id", pyarrow.int64()),
                ("src_dataset_id", pyarrow.int64()),
                ("name", pyarrow.utf8()),
                ("hash", pyarrow.utf8()),
                ("link", pyarrow.utf8()),
                ("frames_count", pyarrow.int32()),
                ("frame_width", pyarrow.int32()),
                ("frame_height", pyarrow.int32()),
                ("frames_to_timecodes", pyarrow.utf8()),
                ("meta", pyarrow.utf8()),
                ("custom_data", pyarrow.utf8()),
                ("created_at", pyarrow.utf8()),
                ("updated_at", pyarrow.utf8()),
                ("ann_json", pyarrow.utf8()),
            ]
        )

        objects_schema = pyarrow.schema(
            [
                ("src_object_id", pyarrow.int64()),
                ("src_video_id", pyarrow.int64()),
                ("class_name", pyarrow.utf8()),
                ("key", pyarrow.utf8()),
                ("tags_json", pyarrow.utf8()),
            ]
        )

        figures_schema = pyarrow.schema(
            [
                ("src_figure_id", pyarrow.int64()),
                ("src_object_id", pyarrow.int64()),
                ("src_video_id", pyarrow.int64()),
                ("frame_index", pyarrow.int32()),
                ("geometry_type", pyarrow.utf8()),
                ("geometry_json", pyarrow.utf8()),
            ]
        )

        if datasets_rows:
            ds_table = pyarrow.Table.from_pylist(datasets_rows, schema=datasets_schema)
            ds_path = os.path.join(payload_dir, "datasets.parquet")
            parquet.write_table(ds_table, ds_path)
            tables_meta.append(
                {
                    "name": "datasets",
                    "path": "datasets.parquet",
                    "row_count": ds_table.num_rows,
                }
            )

        if videos_rows:
            v_table = pyarrow.Table.from_pylist(videos_rows, schema=videos_schema)
            v_path = os.path.join(payload_dir, "videos.parquet")
            parquet.write_table(v_table, v_path)
            tables_meta.append(
                {
                    "name": "videos",
                    "path": "videos.parquet",
                    "row_count": v_table.num_rows,
                }
            )

        if objects_rows:
            o_table = pyarrow.Table.from_pylist(objects_rows, schema=objects_schema)
            o_path = os.path.join(payload_dir, "objects.parquet")
            parquet.write_table(o_table, o_path)
            tables_meta.append(
                {
                    "name": "objects",
                    "path": "objects.parquet",
                    "row_count": o_table.num_rows,
                }
            )

        if figures_rows:
            f_table = pyarrow.Table.from_pylist(figures_rows, schema=figures_schema)
            f_path = os.path.join(payload_dir, "figures.parquet")
            parquet.write_table(f_table, f_path)
            tables_meta.append(
                {
                    "name": "figures",
                    "path": "figures.parquet",
                    "row_count": f_table.num_rows,
                }
            )

        manifest = {
            "schema_version": "video_arrow_v1",
            "tables": tables_meta,
        }
        manifest_path = os.path.join(payload_dir, "manifest.json")
        dump_json_file(manifest, manifest_path)

        # Pack into tar and compress tar.zst in BytesIO
        tar_path = os.path.join(tmp_root, "snapshot.tar")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(payload_dir, arcname=".")

        chunk_size = 1024 * 1024 * 50  # 50 MiB
        outio = io.BytesIO()
        with open(tar_path, "rb") as src:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                outio.write(zstd.compress(chunk))
        outio.seek(0)
        return outio

    finally:
        try:
            clean_dir(tmp_root)
        except Exception:
            pass


def _restore_snapshot(
    api,
    snapshot_bytes: bytes,
    workspace_id: int,
    project_name: Optional[str] = None,
    with_custom_data: bool = True,
    log_progress: bool = True,
    progress_cb=None,
    skip_missed: bool = False,
) -> "ProjectInfo":
    """
    Restore a video project from a snapshot and return ProjectInfo.
    """
    try:
        import pyarrow
        import pyarrow.parquet as parquet
    except Exception as e:
        raise RuntimeError(
            "pyarrow is required to restore video snapshot. Please install pyarrow."
        ) from e

    tmp_root = tempfile.mkdtemp()
    payload_dir = os.path.join(tmp_root, "payload")
    mkdir(payload_dir)

    from supervisely.project.project_type import ProjectType

    try:
        tar_bytes = zstd.decompress(snapshot_bytes)
        tar_path = os.path.join(tmp_root, "snapshot.tar")
        with open(tar_path, "wb") as f:
            f.write(tar_bytes)

        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(payload_dir)

        proj_info_path = os.path.join(payload_dir, "project_info.json")
        proj_meta_path = os.path.join(payload_dir, "project_meta.json")
        key_id_map_path = os.path.join(payload_dir, "key_id_map.json")
        manifest_path = os.path.join(payload_dir, "manifest.json")

        project_info_json = load_json_file(proj_info_path)
        meta_json = load_json_file(proj_meta_path)
        manifest = load_json_file(manifest_path)

        meta = ProjectMeta.from_json(meta_json)
        _ = KeyIdMap().load_json(key_id_map_path)

        if manifest.get("schema_version") != "video_arrow_v1":
            raise RuntimeError(
                f"Unsupported video snapshot schema_version: {manifest.get('schema_version')}"
            )

        src_project_name = project_info_json.get("name")
        if project_name is None:
            project_name = src_project_name

        if api.project.exists(workspace_id, project_name):
            project_name = api.project.get_free_name(workspace_id, project_name)

        project = api.project.create(workspace_id, project_name, ProjectType.VIDEOS)
        new_meta = api.project.update_meta(project.id, meta.to_json())

        if with_custom_data:
            src_custom_data = project_info_json.get("custom_data") or {}
            try:
                api.project.update_custom_data(project.id, src_custom_data, silent=True)
            except Exception:
                logger.warning("Failed to restore project custom_data from snapshot")

        if progress_cb is not None:
            log_progress = False

        # Datasets
        datasets_path = os.path.join(payload_dir, "datasets.parquet")
        if not os.path.exists(datasets_path):
            raise RuntimeError("datasets.parquet is missing in video snapshot")

        ds_table = parquet.read_table(datasets_path)
        ds_rows = ds_table.to_pylist()

        ds_rows.sort(
            key=lambda r: (r["parent_src_dataset_id"] is not None, r["parent_src_dataset_id"])
        )

        dataset_mapping: dict[int, DatasetInfo] = {}
        for row in ds_rows:
            src_ds_id = row["src_dataset_id"]
            parent_src_id = row["parent_src_dataset_id"]
            if parent_src_id is not None:
                parent_ds = dataset_mapping.get(parent_src_id)
                parent_id = parent_ds.id if parent_ds is not None else None
            else:
                parent_id = None

            ds = api.dataset.create(project.id, row["name"], parent_id=parent_id)
            dataset_mapping[src_ds_id] = ds

        # Videos
        videos_path = os.path.join(payload_dir, "videos.parquet")
        if not os.path.exists(videos_path):
            raise RuntimeError("videos.parquet is missing in video snapshot")

        v_table = parquet.read_table(videos_path)
        v_rows = v_table.to_pylist()

        videos_by_dataset: dict[int, List[dict]] = {}
        for row in v_rows:
            src_ds_id = row["src_dataset_id"]
            videos_by_dataset.setdefault(src_ds_id, []).append(row)

        src_to_new_video: dict[int, VideoInfo] = {}

        for src_ds_id, rows in videos_by_dataset.items():
            ds_info = dataset_mapping.get(src_ds_id)
            if ds_info is None:
                logger.warning(
                    f"Dataset with src id={src_ds_id} not found in mapping. "
                    f"Skipping its videos."
                )
                continue

            dataset_id = ds_info.id
            hashed_rows = [r for r in rows if r.get("hash")]
            link_rows = [r for r in rows if not r.get("hash") and r.get("link")]

            ds_progress = progress_cb
            if log_progress and progress_cb is None:
                ds_progress = tqdm_sly(
                    desc=f"Uploading videos to '{ds_info.name}'",
                    total=len(rows),
                )

            if hashed_rows:
                hashes = [r["hash"] for r in hashed_rows]
                names = [r["name"] for r in hashed_rows]
                metas: List[dict] = []
                for r in hashed_rows:
                    meta_dict: dict = {}
                    if r.get("meta"):
                        try:
                            meta_dict.update(json.loads(r["meta"]))
                        except Exception:
                            pass
                    metas.append(meta_dict)

                if skip_missed:
                    existing_hashes = api.video.check_existing_hashes(list(set(hashes)))
                    keep_mask = [h in existing_hashes for h in hashes]
                    if not any(keep_mask):
                        logger.warning(
                            f"All hashed videos for dataset '{ds_info.name}' "
                            f"are missing on server; nothing to upload."
                        )
                    hashes = [h for h, keep in zip(hashes, keep_mask) if keep]
                    names = [n for n, keep in zip(names, keep_mask) if keep]
                    metas = [m for m, keep in zip(metas, keep_mask) if keep]

                if hashes:
                    new_infos = api.video.upload_hashes(
                        dataset_id,
                        names=names,
                        hashes=hashes,
                        metas=metas,
                        progress_cb=ds_progress,
                    )
                    for row, new_info in zip(hashed_rows, new_infos):
                        src_to_new_video[row["src_video_id"]] = new_info
                        if with_custom_data and row.get("custom_data"):
                            try:
                                cd = json.loads(row["custom_data"])
                                api.video.update_custom_data(new_info.id, cd)
                            except Exception:
                                logger.warning(
                                    f"Failed to restore custom_data for video '{new_info.name}'"
                                )

            if link_rows:
                links = [r["link"] for r in link_rows]
                names = [r["name"] for r in link_rows]
                metas: List[dict] = []
                for r in link_rows:
                    meta_dict: dict = {}
                    if r.get("meta"):
                        try:
                            meta_dict.update(json.loads(r["meta"]))
                        except Exception:
                            pass
                    metas.append(meta_dict)

                new_infos_links = api.video.upload_links(
                    dataset_id,
                    links=links,
                    names=names,
                    metas=metas,
                    progress_cb=ds_progress,
                )
                for row, new_info in zip(link_rows, new_infos_links):
                    src_to_new_video[row["src_video_id"]] = new_info
                    if with_custom_data and row.get("custom_data"):
                        try:
                            cd = json.loads(row["custom_data"])
                            api.video.update_custom_data(new_info.id, cd)
                        except Exception:
                            logger.warning(
                                f"Failed to restore custom_data for video '{new_info.name}'"
                            )

            if ds_progress is not None:
                ds_progress(len(rows))

        # Annotations
        ann_temp_dir = os.path.join(tmp_root, "anns")
        mkdir(ann_temp_dir)

        anns_by_dataset: dict[int, List[Tuple[int, str]]] = {}
        for row in v_rows:
            src_vid = row["src_video_id"]
            new_info = src_to_new_video.get(src_vid)
            if new_info is None:
                continue
            src_ds_id = row["src_dataset_id"]
            anns_by_dataset.setdefault(src_ds_id, []).append((new_info.id, row["ann_json"]))

        for src_ds_id, items in anns_by_dataset.items():
            ds_info = dataset_mapping.get(src_ds_id)
            if ds_info is None:
                continue

            video_ids: List[int] = []
            ann_paths: List[str] = []

            for vid_id, ann_json_str in items:
                video_ids.append(vid_id)
                ann_path = os.path.join(ann_temp_dir, f"{vid_id}.json")
                try:
                    parsed = json.loads(ann_json_str)
                except Exception:
                    logger.warning(
                        f"Failed to parse ann_json for restored video id={vid_id}, "
                        f"skipping its annotation."
                    )
                    continue
                dump_json_file(parsed, ann_path)
                ann_paths.append(ann_path)

            if not video_ids:
                continue

            anns_progress = progress_cb
            if log_progress and progress_cb is None:
                anns_progress = tqdm_sly(
                    desc=f"Uploading annotations to '{ds_info.name}'",
                    total=len(video_ids),
                    leave=False,
                )
            for vid_id, ann_path in zip(video_ids, ann_paths):
                try:
                    ann_json = load_json_file(ann_path)
                    ann = VideoAnnotation.from_json(
                        ann_json,
                        new_meta,
                        key_id_map=KeyIdMap(),
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to deserialize annotation for restored video id={vid_id}: {e}"
                    )
                    continue

                api.video.annotation.append(vid_id, ann)
                if anns_progress is not None:
                    anns_progress(1)

        return project

    finally:
        try:
            clean_dir(tmp_root)
        except Exception:
            pass
