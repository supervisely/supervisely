from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Union

from tqdm import tqdm

from supervisely._utils import batched, logger
from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.module_api import ApiField
from supervisely.api.project_api import ProjectInfo
from supervisely.api.video.video_api import VideoInfo
from supervisely.io.fs import mkdir
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project import Dataset
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.task.progress import tqdm_sly
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_annotation import VideoAnnotation

from supervisely.project.versioning.base_snapshot import BaseSnapshotHandler
from supervisely.project.versioning.utils import _require_pyarrow


DEFAULT_VIDEO_SNAPSHOT_SCHEMA_VERSION = "v2.0.0"


class _VideoVersioningV1:
    """
    Video snapshot format v1 (schema_version="v2.0.0").

    Layout inside payload dir:
      - project_info.json
      - project_meta.json
      - key_id_map.json
      - manifest.json
      - datasets.parquet
      - videos.parquet
      - objects.parquet
      - figures.parquet
    """

    schema_version = DEFAULT_VIDEO_SNAPSHOT_SCHEMA_VERSION

    def build_payload(
        self,
        api: Api,
        project_id: int,
        payload_dir: str,
        dataset_ids: Optional[List[int]] = None,
        batch_size: int = 50,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        pa, pq = _require_pyarrow()

        project_info = api.project.get_info_by_id(project_id)
        meta = ProjectMeta.from_json(api.project.get_meta(project_id, with_settings=True))
        key_id_map = KeyIdMap()

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

        # api.dataset.tree() doesn't include custom_data
        ds_custom_data_by_id: dict[int, dict] = {}
        try:
            for ds in api.dataset.get_list(project_id, recursive=True, include_custom_data=True):
                if getattr(ds, "custom_data", None) is not None:
                    ds_custom_data_by_id[ds.id] = ds.custom_data
        except Exception:
            ds_custom_data_by_id = {}

        for parents, ds_info in api.dataset.tree(project_id):
            if dataset_ids_filter is not None and ds_info.id not in dataset_ids_filter:
                continue

            full_path = Dataset._get_dataset_path(ds_info.name, parents)
            ds_custom_data = ds_custom_data_by_id.get(ds_info.id)
            datasets_rows.append(
                {
                    "src_dataset_id": ds_info.id,
                    "parent_src_dataset_id": ds_info.parent_id,
                    "name": ds_info.name,
                    "full_path": full_path,
                    "description": ds_info.description,
                    "custom_data": (
                        json.dumps(ds_custom_data)
                        if isinstance(ds_custom_data, dict) and len(ds_custom_data) > 0
                        else None
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
        datasets_schema = pa.schema(
            [
                ("src_dataset_id", pa.int64()),
                ("parent_src_dataset_id", pa.int64()),
                ("name", pa.utf8()),
                ("full_path", pa.utf8()),
                ("description", pa.utf8()),
                ("custom_data", pa.utf8()),
            ]
        )

        videos_schema = pa.schema(
            [
                ("src_video_id", pa.int64()),
                ("src_dataset_id", pa.int64()),
                ("name", pa.utf8()),
                ("hash", pa.utf8()),
                ("link", pa.utf8()),
                ("frames_count", pa.int32()),
                ("frame_width", pa.int32()),
                ("frame_height", pa.int32()),
                ("frames_to_timecodes", pa.utf8()),
                ("meta", pa.utf8()),
                ("custom_data", pa.utf8()),
                ("created_at", pa.utf8()),
                ("updated_at", pa.utf8()),
                ("ann_json", pa.utf8()),
            ]
        )

        objects_schema = pa.schema(
            [
                ("src_object_id", pa.int64()),
                ("src_video_id", pa.int64()),
                ("class_name", pa.utf8()),
                ("key", pa.utf8()),
                ("tags_json", pa.utf8()),
            ]
        )

        figures_schema = pa.schema(
            [
                ("src_figure_id", pa.int64()),
                ("src_object_id", pa.int64()),
                ("src_video_id", pa.int64()),
                ("frame_index", pa.int32()),
                ("geometry_type", pa.utf8()),
                ("geometry_json", pa.utf8()),
            ]
        )

        if datasets_rows:
            ds_table = pa.Table.from_pylist(datasets_rows, schema=datasets_schema)
            ds_path = os.path.join(payload_dir, "datasets.parquet")
            pq.write_table(ds_table, ds_path)
            tables_meta.append(
                {"name": "datasets", "path": "datasets.parquet", "row_count": ds_table.num_rows}
            )

        if videos_rows:
            v_table = pa.Table.from_pylist(videos_rows, schema=videos_schema)
            v_path = os.path.join(payload_dir, "videos.parquet")
            pq.write_table(v_table, v_path)
            tables_meta.append(
                {"name": "videos", "path": "videos.parquet", "row_count": v_table.num_rows}
            )

        if objects_rows:
            o_table = pa.Table.from_pylist(objects_rows, schema=objects_schema)
            o_path = os.path.join(payload_dir, "objects.parquet")
            pq.write_table(o_table, o_path)
            tables_meta.append(
                {"name": "objects", "path": "objects.parquet", "row_count": o_table.num_rows}
            )

        if figures_rows:
            f_table = pa.Table.from_pylist(figures_rows, schema=figures_schema)
            f_path = os.path.join(payload_dir, "figures.parquet")
            pq.write_table(f_table, f_path)
            tables_meta.append(
                {"name": "figures", "path": "figures.parquet", "row_count": f_table.num_rows}
            )

        manifest = {"schema_version": self.schema_version, "tables": tables_meta}
        manifest_path = os.path.join(payload_dir, "manifest.json")
        dump_json_file(manifest, manifest_path)

    def restore_payload(
        self,
        api: Api,
        payload_dir: str,
        workspace_id: int,
        project_name: Optional[str] = None,
        with_custom_data: bool = True,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_missed: bool = False,
    ) -> ProjectInfo:
        _, pq = _require_pyarrow()

        proj_info_path = os.path.join(payload_dir, "project_info.json")
        proj_meta_path = os.path.join(payload_dir, "project_meta.json")
        key_id_map_path = os.path.join(payload_dir, "key_id_map.json")
        manifest_path = os.path.join(payload_dir, "manifest.json")

        project_info_json = load_json_file(proj_info_path)
        meta_json = load_json_file(proj_meta_path)
        manifest = load_json_file(manifest_path)

        meta = ProjectMeta.from_json(meta_json)
        _ = KeyIdMap().load_json(key_id_map_path)

        if manifest.get("schema_version") != self.schema_version:
            raise RuntimeError(
                f"Unsupported video snapshot schema_version: {manifest.get('schema_version')}"
            )

        src_project_name = project_info_json.get("name")
        src_project_desc = project_info_json.get("description")
        if project_name is None:
            project_name = src_project_name

        if api.project.exists(workspace_id, project_name):
            project_name = api.project.get_free_name(workspace_id, project_name)

        project = api.project.create(workspace_id, project_name, ProjectType.VIDEOS, src_project_desc)
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
        ds_rows = []
        datasets_path = os.path.join(payload_dir, "datasets.parquet")
        if os.path.exists(datasets_path):
            ds_table = pq.read_table(datasets_path)
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

            custom_data = None
            if with_custom_data:
                raw_cd = row.get("custom_data")
                if isinstance(raw_cd, str) and raw_cd.strip():
                    try:
                        custom_data = json.loads(raw_cd)
                    except Exception:
                        logger.warning(
                            f"Failed to parse dataset custom_data for '{row.get('name')}', skipping it."
                        )
                elif isinstance(raw_cd, dict):
                    custom_data = raw_cd

            ds = api.dataset.create(
                project.id,
                name=row["name"],
                description=row["description"],
                parent_id=parent_id,
                custom_data=custom_data,
            )
            if with_custom_data and custom_data is not None:
                try:
                    api.dataset.update_custom_data(ds.id, custom_data)
                except Exception:
                    logger.warning(f"Failed to restore custom_data for dataset '{row.get('name')}'")
            dataset_mapping[src_ds_id] = ds

        # Videos
        v_rows = []
        videos_path = os.path.join(payload_dir, "videos.parquet")
        if os.path.exists(videos_path):
            v_table = pq.read_table(videos_path)
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
                    f"Dataset with src id={src_ds_id} not found in mapping. Skipping its videos."
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
                if skip_missed:
                    existing_hashes = api.video.check_existing_hashes(
                        list({r["hash"] for r in hashed_rows})
                    )
                    kept_hashed_rows = [r for r in hashed_rows if r["hash"] in existing_hashes]
                    if not kept_hashed_rows:
                        logger.warning(
                            f"All hashed videos for dataset '{ds_info.name}' "
                            f"are missing on server; nothing to upload."
                        )
                    hashed_rows = kept_hashed_rows

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
        ann_temp_dir = os.path.join(os.path.dirname(payload_dir), "anns")
        mkdir(ann_temp_dir)

        anns_by_dataset: dict[int, List[tuple[int, str]]] = {}
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
                        f"Failed to parse ann_json for restored video id={vid_id}, skipping its annotation."
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


_VIDEO_SNAPSHOT_HANDLERS: Dict[str, BaseSnapshotHandler] = {
    DEFAULT_VIDEO_SNAPSHOT_SCHEMA_VERSION: _VideoVersioningV1(),
}


def get_video_snapshot_handler(schema_version: str) -> BaseSnapshotHandler:
    handler = _VIDEO_SNAPSHOT_HANDLERS.get(schema_version)
    if handler is None:
        raise RuntimeError(f"Unsupported video snapshot schema_version: {schema_version!r}")
    return handler


