from __future__ import annotations

import io
import json
import struct
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

import supervisely as sly
import supervisely.volume_annotation.constants as volume_constants
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely.geometry.mask_3d import Mask3D

from supervisely.project.versioning.utils import _require_pyarrow


DEFAULT_VOLUME_SNAPSHOT_SCHEMA_VERSION = "v2.0.0"
_SCHEMA_VERSION_FIELD = "__schema_version__"

_SERIALIZATION_MAGIC = b"SLYVOLPAR"
_SERIALIZATION_VERSION = 1

_SECTION_PROJECT_INFO = 1
_SECTION_PROJECT_META = 2
_SECTION_DATASETS = 3
_SECTION_VOLUMES = 4
_SECTION_ANNOTATIONS = 5


def _json_dumps(data) -> str:
    if isinstance(data, str):
        return data
    return json.dumps(data, ensure_ascii=False)


def _json_bytes(data) -> bytes:
    return _json_dumps(data).encode("utf-8")


def _build_table(pa_module, columns: Dict[str, Tuple[List, Any]]):
    arrays = {}
    for name, (values, dtype) in columns.items():
        arrays[name] = pa_module.array(values, type=dtype)
    return pa_module.table(arrays)


def _table_to_parquet_bytes(pa_module, table) -> bytes:
    from pyarrow import parquet as pq

    sink = pa_module.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def _parquet_bytes_to_table(pa_module, data: bytes):
    if not data:
        return pa_module.table({})
    from pyarrow import parquet as pq

    buffer = pa_module.BufferReader(data)
    return pq.read_table(buffer)


def _assemble_sections(sections: List[Tuple[int, bytes]]) -> bytes:
    if len(sections) > 255:
        raise RuntimeError("Too many sections for VolumeProject binary payload")
    buffer = io.BytesIO()
    buffer.write(_SERIALIZATION_MAGIC)
    buffer.write(struct.pack(">B", _SERIALIZATION_VERSION))
    buffer.write(struct.pack(">B", len(sections)))
    for section_type, payload in sections:
        if payload is None:
            payload = b""
        buffer.write(struct.pack(">B", section_type))
        buffer.write(struct.pack(">Q", len(payload)))
        buffer.write(payload)
    return buffer.getvalue()


def _parse_parquet_sections(raw_data: Union[bytes, memoryview]) -> Dict[int, bytes]:
    view = raw_data if isinstance(raw_data, memoryview) else memoryview(raw_data)
    header_len = len(_SERIALIZATION_MAGIC) + 2
    if len(view) < header_len:
        logger.warning(
            f"VolumeProject binary payload too small: {len(view)} bytes "
            f"(need >= {header_len}). First bytes(hex)={view[: min(len(view), 16)].tobytes().hex()}",
        )
        raise RuntimeError("Corrupted VolumeProject binary payload")

    if view[: len(_SERIALIZATION_MAGIC)].tobytes() != _SERIALIZATION_MAGIC:
        found = view[: len(_SERIALIZATION_MAGIC)].tobytes()
        logger.warning(
            f"VolumeProject binary payload magic mismatch. expected={_SERIALIZATION_MAGIC.hex()} "
            f"found={found.hex()} total_bytes={len(view)} prefix16(hex)={view[:16].tobytes().hex()}",
        )
        raise RuntimeError("Unsupported VolumeProject binary payload format (magic mismatch).")

    offset = len(_SERIALIZATION_MAGIC)
    version = view[offset]
    offset += 1
    if version != _SERIALIZATION_VERSION:
        logger.warning(
            "VolumeProject binary payload version mismatch. expected=%d found=%d total_bytes=%d",
            _SERIALIZATION_VERSION,
            version,
            len(view),
        )
        raise RuntimeError(f"Unsupported VolumeProject binary payload version: {version}")

    section_count = view[offset]
    offset += 1
    sections: Dict[int, bytes] = {}
    for _ in range(section_count):
        if offset + 9 > len(view):
            raise RuntimeError("Corrupted VolumeProject binary payload")
        section_type = view[offset]
        offset += 1
        length = int.from_bytes(view[offset : offset + 8], "big")
        offset += 8
        if offset + length > len(view):
            raise RuntimeError("Corrupted VolumeProject binary payload")
        sections[section_type] = view[offset : offset + length].tobytes()
        offset += length
    return sections


def _serialize_payload_to_parquet_blob(pa_module, payload: Dict[str, Dict]) -> bytes:
    dataset_records: List[Dict] = payload.get("dataset_infos", []) or []
    volume_records: List[Dict] = payload.get("volume_infos", []) or []
    annotations_dict: Dict[str, Dict] = payload.get("annotations", {}) or {}

    dataset_table = _build_table(
        pa_module,
        {
            "dataset_id": ([record.get("id") for record in dataset_records], pa_module.int64()),
            "json": (
                [_json_dumps(record) for record in dataset_records],
                pa_module.large_string(),
            ),
        },
    )

    volume_table = _build_table(
        pa_module,
        {
            "volume_id": ([record.get("id") for record in volume_records], pa_module.int64()),
            "dataset_id": (
                [record.get("dataset_id") for record in volume_records],
                pa_module.int64(),
            ),
            "json": (
                [_json_dumps(record) for record in volume_records],
                pa_module.large_string(),
            ),
        },
    )

    ann_volume_ids: List[Optional[int]] = []
    ann_payloads: List[str] = []
    for volume_id_str, ann in annotations_dict.items():
        try:
            volume_id = int(volume_id_str)
        except (TypeError, ValueError):
            volume_id = None
        ann_volume_ids.append(volume_id)
        ann_payloads.append(_json_dumps(ann))

    annotations_table = _build_table(
        pa_module,
        {
            "volume_id": (ann_volume_ids, pa_module.int64()),
            "annotation": (ann_payloads, pa_module.large_string()),
        },
    )

    sections = [
        (_SECTION_PROJECT_INFO, _json_bytes(payload.get("project_info", {}))),
        (_SECTION_PROJECT_META, _json_bytes(payload.get("project_meta", {}))),
        (_SECTION_DATASETS, _table_to_parquet_bytes(pa_module, dataset_table)),
        (_SECTION_VOLUMES, _table_to_parquet_bytes(pa_module, volume_table)),
        (_SECTION_ANNOTATIONS, _table_to_parquet_bytes(pa_module, annotations_table)),
    ]
    return _assemble_sections(sections)


def _deserialize_payload_from_parquet(pa_module, raw_data: Union[bytes, memoryview]) -> Dict:
    sections = _parse_parquet_sections(raw_data)

    try:
        project_info = json.loads(sections[_SECTION_PROJECT_INFO].decode("utf-8"))
        project_meta = json.loads(sections[_SECTION_PROJECT_META].decode("utf-8"))
    except KeyError as exc:
        raise RuntimeError("VolumeProject payload missing metadata section") from exc

    if _SECTION_DATASETS not in sections:
        logger.warning("VolumeProject blob has no datasets section; treating as empty.")
    if _SECTION_VOLUMES not in sections:
        logger.warning("VolumeProject blob has no volumes section; treating as empty.")
    if _SECTION_ANNOTATIONS not in sections:
        logger.warning("VolumeProject blob has no annotations section; treating as empty.")

    dataset_table = _parquet_bytes_to_table(pa_module, sections.get(_SECTION_DATASETS, b""))
    volume_table = _parquet_bytes_to_table(pa_module, sections.get(_SECTION_VOLUMES, b""))
    annotations_table = _parquet_bytes_to_table(pa_module, sections.get(_SECTION_ANNOTATIONS, b""))

    dataset_records: List[Dict] = []
    if dataset_table is not None and dataset_table.num_rows and "json" in dataset_table.column_names:
        dataset_jsons = dataset_table.column("json").to_pylist()
        dataset_records = [json.loads(item) for item in dataset_jsons]

    volume_records: List[Dict] = []
    if volume_table is not None and volume_table.num_rows and "json" in volume_table.column_names:
        volume_jsons = volume_table.column("json").to_pylist()
        volume_records = [json.loads(item) for item in volume_jsons]

    annotations: Dict[str, Dict] = {}
    if (
        annotations_table is not None
        and annotations_table.num_rows
        and "volume_id" in annotations_table.column_names
        and "annotation" in annotations_table.column_names
    ):
        annotation_ids = annotations_table.column("volume_id").to_pylist()
        annotation_payloads = annotations_table.column("annotation").to_pylist()
        for volume_id, annotation_json in zip(annotation_ids, annotation_payloads):
            if volume_id is None:
                continue
            annotations[str(volume_id)] = json.loads(annotation_json)

    return {
        "project_info": project_info,
        "project_meta": project_meta,
        "dataset_infos": dataset_records,
        "volume_infos": volume_records,
        "annotations": annotations,
    }


def _load_mask_geometries(api: Api, ann: VolumeAnnotation, key_id_map: KeyIdMap) -> None:
    for sf in ann.spatial_figures:
        if sf.geometry.name() != Mask3D.name():
            continue
        api.volume.figure.load_sf_geometry(sf, key_id_map)


def _detect_schema_version(raw_data: Union[bytes, memoryview]) -> str:
    try:
        sections = _parse_parquet_sections(raw_data)
        info = json.loads(sections[_SECTION_PROJECT_INFO].decode("utf-8"))
        return info.get(_SCHEMA_VERSION_FIELD) or DEFAULT_VOLUME_SNAPSHOT_SCHEMA_VERSION
    except Exception:
        return DEFAULT_VOLUME_SNAPSHOT_SCHEMA_VERSION


class _VolumeVersioningV1:
    schema_version = DEFAULT_VOLUME_SNAPSHOT_SCHEMA_VERSION

    def build_blob(
        self,
        api: Api,
        project_id: int,
        dataset_ids: Optional[List[int]] = None,
        download_volumes: bool = True,
        log_progress: bool = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> bytes:
        pa, _ = _require_pyarrow()

        ds_filters = (
            [{"field": "id", "operator": "in", "value": dataset_ids}]
            if dataset_ids is not None
            else None
        )

        project_info = api.project.get_info_by_id(project_id)
        project_meta = api.project.get_meta(project_id, with_settings=True)
        project_meta_obj = ProjectMeta.from_json(project_meta)
        dataset_infos = api.dataset.get_list(project_id, filters=ds_filters, recursive=True)

        dataset_records = [dataset_info._asdict() for dataset_info in dataset_infos]
        volume_records: List[Dict] = []
        annotations: Dict[str, Dict] = {}
        key_id_map = KeyIdMap()

        for dataset_info in dataset_infos:
            if dataset_ids is not None and dataset_info.id not in dataset_ids:
                continue

            volumes = api.volume.get_list(dataset_info.id)
            if len(volumes) == 0:
                continue

            if not download_volumes:
                continue

            ds_progress = progress_cb
            if log_progress and progress_cb is None:
                ds_progress = tqdm_sly(
                    desc="Collecting volumes from: {!r}".format(dataset_info.name),
                    total=len(volumes),
                )

            volume_ids = [volume_info.id for volume_info in volumes]
            ann_jsons = api.volume.annotation.download_bulk(dataset_info.id, volume_ids)

            ann_by_volume_id: Dict[int, Dict[str, Any]] = {}
            spatial_figures_by_volume: Dict[int, Dict[int, Dict[str, Any]]] = {}
            for ann_json in ann_jsons:
                volume_id = ann_json.get(ApiField.VOLUME_ID)
                if volume_id is None:
                    continue
                ann_by_volume_id[volume_id] = ann_json
                figures_list = ann_json.get(volume_constants.SPATIAL_FIGURES, []) or []
                fig_id_to_spatial_figure: Dict[int, Dict[str, Any]] = {}
                for spatial_figure in figures_list:
                    fig_id = spatial_figure.get("id")
                    if fig_id is not None:
                        fig_id_to_spatial_figure[fig_id] = spatial_figure
                spatial_figures_by_volume[volume_id] = fig_id_to_spatial_figure

            figures_dict = api.volume.figure.download(dataset_info.id, volume_ids)
            for volume_id, figure_infos in figures_dict.items():
                ann_json = ann_by_volume_id.get(volume_id)
                if ann_json is None:
                    continue
                fig_id_to_spatial_figure = spatial_figures_by_volume.get(volume_id, {})
                for figure_info in figure_infos:
                    spatial_figure = fig_id_to_spatial_figure.get(figure_info.id)
                    if spatial_figure is not None:
                        spatial_figure[ApiField.CUSTOM_DATA] = figure_info.custom_data

            for volume_info, ann_json in zip(volumes, ann_jsons):
                ann = VolumeAnnotation.from_json(ann_json, project_meta_obj, key_id_map)
                _load_mask_geometries(api, ann, key_id_map)
                volume_records.append(volume_info._asdict())
                annotations[str(volume_info.id)] = ann.to_json()
                if progress_cb is not None:
                    progress_cb(1)
                if ds_progress is not None:
                    ds_progress(1)

        project_info_dict = project_info._asdict()
        project_info_dict[_SCHEMA_VERSION_FIELD] = self.schema_version

        payload = {
            "project_info": project_info_dict,
            "project_meta": project_meta,
            "dataset_infos": dataset_records,
            "volume_infos": volume_records,
            "annotations": annotations,
        }
        return _serialize_payload_to_parquet_blob(pa, payload)

    def restore_blob(
        self,
        api: Api,
        raw_data: Union[bytes, memoryview],
        workspace_id: int,
        project_name: Optional[str] = None,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_missed_entities: bool = False,
    ):
        pa, _ = _require_pyarrow()
        payload = _deserialize_payload_from_parquet(pa, raw_data)

        project_meta = ProjectMeta.from_json(payload["project_meta"])
        dataset_records: List[Dict] = payload.get("dataset_infos", [])
        volume_records: List[Dict] = payload.get("volume_infos", [])
        annotations: Dict[str, Dict] = payload.get("annotations", {})

        project_title = project_name or payload["project_info"].get("name")
        if api.project.exists(workspace_id, project_title):
            project_title = api.project.get_free_name(workspace_id, project_title)
        new_project_info = api.project.create(workspace_id, project_title, ProjectType.VOLUMES)
        api.project.update_meta(new_project_info.id, project_meta)

        custom_data = new_project_info.custom_data
        source_project_id = payload["project_info"].get("id")
        version_info = payload["project_info"].get("version") or {}
        custom_data["restored_from"] = {
            "project_id": source_project_id,
            "version_num": version_info.get("version"),
        }
        original_custom_data = payload["project_info"].get("custom_data") or {}
        custom_data.update(original_custom_data)
        api.project.update_custom_data(new_project_info.id, custom_data, silent=True)

        dataset_mapping: Dict[int, sly.DatasetInfo] = {}
        sorted_datasets = sorted(
            dataset_records,
            key=lambda data: (data.get("parent_id") is not None, data.get("parent_id") or 0),
        )
        for dataset_data in sorted_datasets:
            parent_ds_info = dataset_mapping.get(dataset_data.get("parent_id"))
            new_parent_id = parent_ds_info.id if parent_ds_info else None
            new_dataset_info = api.dataset.create(
                project_id=new_project_info.id,
                name=dataset_data.get("name"),
                description=dataset_data.get("description"),
                parent_id=new_parent_id,
                custom_data=dataset_data.get("custom_data"),
            )
            dataset_mapping[dataset_data.get("id")] = new_dataset_info

        volume_mapping: Dict[int, sly.VolumeInfo] = {}
        volumes_by_dataset: Dict[int, List[Dict]] = defaultdict(list)
        for volume_data in volume_records:
            volumes_by_dataset[volume_data.get("dataset_id")].append(volume_data)

        for old_dataset_id, dataset_volumes in volumes_by_dataset.items():
            new_dataset_info = dataset_mapping.get(old_dataset_id)
            if new_dataset_info is None:
                continue

            missing_hash_volumes = [vol for vol in dataset_volumes if not vol.get("hash")]
            if missing_hash_volumes:
                missing_names = [vol.get("name") or str(vol.get("id")) for vol in missing_hash_volumes]
                if skip_missed_entities:
                    for vol_name in missing_names:
                        logger.warning(
                            "Volume %r skipped during restoration because its source hash is unavailable.",
                            vol_name,
                        )
                    dataset_volumes_to_upload = [vol for vol in dataset_volumes if vol.get("hash")]
                    if len(dataset_volumes_to_upload) == 0:
                        continue
                else:
                    raise RuntimeError(
                        "Cannot restore volumes without available hash. Missing volume names: {}".format(
                            ", ".join(missing_names)
                        )
                    )
            else:
                dataset_volumes_to_upload = list(dataset_volumes)

            hashes = [volume.get("hash") for volume in dataset_volumes_to_upload]
            names = [volume.get("name") for volume in dataset_volumes_to_upload]
            metas = [volume.get("meta") for volume in dataset_volumes_to_upload]

            ds_progress = progress_cb
            if log_progress and progress_cb is None:
                ds_progress = tqdm_sly(
                    desc="Uploading volumes to {!r}".format(new_dataset_info.name),
                    total=len(dataset_volumes_to_upload),
                )

            new_volume_infos = api.volume.upload_hashes(
                new_dataset_info.id,
                names=names,
                hashes=hashes,
                metas=metas,
                progress_cb=ds_progress,
            )

            for old_volume, new_volume in zip(dataset_volumes_to_upload, new_volume_infos):
                volume_mapping[old_volume.get("id")] = new_volume

        for volume_id_str, ann_json in annotations.items():
            new_volume_info = volume_mapping.get(int(volume_id_str))
            if new_volume_info is None:
                if skip_missed_entities:
                    logger.warning(
                        "Annotation for volume %s skipped because the source volume was not restored.",
                        volume_id_str,
                    )
                continue
            ann_json["volumeId"] = new_volume_info.id
            ann = VolumeAnnotation.from_json(ann_json, project_meta, None)
            api.volume.annotation.append(new_volume_info.id, ann, None)

        return api.project.get_info_by_id(new_project_info.id)


_VOLUME_SNAPSHOT_HANDLERS: Dict[str, _VolumeVersioningV1] = {
    DEFAULT_VOLUME_SNAPSHOT_SCHEMA_VERSION: _VolumeVersioningV1(),
}


def get_volume_snapshot_handler(schema_version: str) -> _VolumeVersioningV1:
    handler = _VOLUME_SNAPSHOT_HANDLERS.get(schema_version)
    if handler is None:
        raise RuntimeError(f"Unsupported volume snapshot schema_version: {schema_version!r}")
    return handler


def get_volume_snapshot_handler_from_blob(raw_data: Union[bytes, memoryview]) -> _VolumeVersioningV1:
    return get_volume_snapshot_handler(_detect_schema_version(raw_data))


