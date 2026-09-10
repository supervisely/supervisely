"""Reading the Parquet tables of an image project snapshot (schema v2.0.0).

Two consumers, one place: a restore needs the whole payload as the in-memory tuple the
pickle format used to unpickle, while a comparison between two snapshots reads columns
in batches and never materializes the project. Both go through here so the column names
are written down once.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from supervisely.project.versioning.schema_fields import VersionSchemaField
from supervisely.project.versioning.tag_schema import OWNER_FIGURE, OWNER_ITEM

DATASETS_TABLE = "datasets"
IMAGES_TABLE = "images"
FIGURES_TABLE = "figures"
ALPHA_GEOMETRIES_TABLE = "alpha_geometries"
TAGS_TABLE = "tags"

PROJECT_INFO_FILE = "project_info.json"
PROJECT_META_FILE = "project_meta.json"


PYARROW_REQUIRED = (
    "pyarrow is required for Parquet project snapshots. "
    "Install it with: pip install supervisely[versioning]"
)


def import_pyarrow():
    """pyarrow and its parquet module, with the install hint the SDK's extras use."""
    try:
        import pyarrow  # pylint: disable=import-error
        import pyarrow.parquet as parquet  # pylint: disable=import-error

        return pyarrow, parquet
    except Exception as e:
        raise RuntimeError(PYARROW_REQUIRED) from e


def import_parquet():
    return import_pyarrow()[1]


def table_path(payload_dir: str, table: str) -> Optional[str]:
    """Path of a snapshot table, or None when the snapshot has no rows for it.

    An empty table is written as no file at all, so a missing one is normal and means
    "nothing of this kind in the project", not a corrupt snapshot.
    """
    path = os.path.join(payload_dir, f"{table}.parquet")
    return path if os.path.isfile(path) else None


def existing_columns(source, columns: Optional[Sequence[str]]) -> Optional[List[str]]:
    """``columns`` narrowed to those the file actually has.

    Schemas gain columns between versions, so a reader asking for a column by name has
    to tolerate an older snapshot that predates it - Parquet raises on an unknown column
    rather than returning nulls.
    """
    if columns is None:
        return None
    parquet = import_parquet()
    present = set(parquet.ParquetFile(source).schema_arrow.names)
    return [c for c in columns if c in present] or None


def iter_rows_from_source(
    source,
    columns: Optional[Sequence[str]] = None,
    batch_size: int = 5000,
) -> Iterator[List[dict]]:
    """Yield lists of rows from any Parquet source - a path, or bytes in memory.

    Volume snapshots keep their tables inside one bespoke binary blob rather than as
    files in a payload directory, so the reader has to be able to work off a buffer too.
    """
    parquet = import_parquet()
    columns = existing_columns(source, columns)
    parquet_file = parquet.ParquetFile(source)
    try:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=list(columns or []) or None
        ):
            yield batch.to_pylist()
    finally:
        # A caller is free to stop reading early, so the handle is released here rather
        # than at exhaustion.
        parquet_file.close()


def iter_rows(
    payload_dir: str,
    table: str,
    columns: Optional[Sequence[str]] = None,
    batch_size: int = 5000,
) -> Iterator[List[dict]]:
    """Yield lists of rows from one snapshot table, one row group batch at a time.

    ``columns`` is the whole point of the format: reading four columns of a figures
    table costs four columns, not the table.
    """
    path = table_path(payload_dir, table)
    if path is None:
        return
    for batch in iter_rows_from_source(path, columns=columns, batch_size=batch_size):
        yield batch


def read_all_rows(
    payload_dir: str, table: str, columns: Optional[Sequence[str]] = None
) -> List[dict]:
    rows: List[dict] = []
    for batch in iter_rows(payload_dir, table, columns=columns):
        rows.extend(batch)
    return rows


def loads_or(value, default):
    """Parse a JSON column, falling back to ``default`` for nulls and unreadable values."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return default
    return default if parsed is None else parsed


def namedtuple_from_dict(cls, data: Dict[str, Any]):
    """Build a NamedTuple from a field->value mapping, tolerating drift in both directions.

    Keys the class does not have are dropped and fields the mapping does not have become
    None. This is the same problem the pickle format needed ``CustomUnpickler`` and
    ``restore_legacy_defaults`` for, except that here the snapshot stores plain columns,
    so it is four lines instead of a class-shadowing unpickler.
    """
    return cls(**{field: data.get(field) for field in cls._fields})


def build_dataset_info(row: dict):
    from supervisely.api.dataset_api import DatasetInfo

    return namedtuple_from_dict(
        DatasetInfo,
        {
            "id": row[VersionSchemaField.SRC_DATASET_ID],
            "name": row[VersionSchemaField.NAME],
            "description": row.get(VersionSchemaField.DESCRIPTION),
            "parent_id": row.get(VersionSchemaField.PARENT_SRC_DATASET_ID),
            "custom_data": loads_or(row.get(VersionSchemaField.CUSTOM_DATA), {}),
        },
    )


def build_image_info(row: dict, tags: Optional[List[dict]] = None):
    from supervisely.api.image_api import ImageInfo

    return namedtuple_from_dict(
        ImageInfo,
        {
            "id": row[VersionSchemaField.SRC_IMAGE_ID],
            "dataset_id": row[VersionSchemaField.SRC_DATASET_ID],
            "parent_id": row.get(VersionSchemaField.PARENT_SRC_IMAGE_ID),
            "name": row[VersionSchemaField.NAME],
            "hash": row.get(VersionSchemaField.HASH),
            "link": row.get(VersionSchemaField.LINK),
            "mime": row.get(VersionSchemaField.MIME),
            "ext": row.get(VersionSchemaField.EXT),
            "size": row.get(VersionSchemaField.SIZE),
            "width": row.get(VersionSchemaField.WIDTH),
            "height": row.get(VersionSchemaField.HEIGHT),
            "created_at": row.get(VersionSchemaField.CREATED_AT),
            "updated_at": row.get(VersionSchemaField.UPDATED_AT),
            "created_by": row.get(VersionSchemaField.CREATED_BY_ID),
            "description": row.get(VersionSchemaField.DESCRIPTION),
            # Normalized rather than left null: an upload passes meta straight to the
            # API and iterates tags, so the empty forms have to be the real ones.
            "meta": loads_or(row.get(VersionSchemaField.META), {}),
            # Tags are rows in the shared tags table, joined back on for the restore.
            "tags": tags or [],
            "related_data_id": row.get(VersionSchemaField.RELATED_DATA_ID),
            "download_id": row.get(VersionSchemaField.DOWNLOAD_ID),
            "offset_start": row.get(VersionSchemaField.OFFSET_START),
            "offset_end": row.get(VersionSchemaField.OFFSET_END),
        },
    )


def build_figure_info(
    row: dict, alpha_geometry: Optional[dict] = None, tags: Optional[List[dict]] = None
):
    from supervisely.api.entity_annotation.figure_api import FigureInfo

    geometry = loads_or(row.get(VersionSchemaField.GEOMETRY_JSON), None)
    if geometry is None and alpha_geometry is not None:
        geometry = alpha_geometry
    return namedtuple_from_dict(
        FigureInfo,
        {
            "id": row[VersionSchemaField.SRC_FIGURE_ID],
            "entity_id": row[VersionSchemaField.SRC_IMAGE_ID],
            "dataset_id": row.get(VersionSchemaField.SRC_DATASET_ID),
            "class_id": row.get(VersionSchemaField.CLASS_ID),
            "created_at": row.get(VersionSchemaField.CREATED_AT),
            "updated_at": row.get(VersionSchemaField.UPDATED_AT),
            "geometry_type": row.get(VersionSchemaField.GEOMETRY_TYPE),
            "geometry": geometry,
            "geometry_meta": loads_or(row.get(VersionSchemaField.GEOMETRY_META_JSON), {}),
            "meta": loads_or(row.get(VersionSchemaField.META), {}),
            "tags": tags or [],
            "priority": row.get(VersionSchemaField.PRIORITY),
            "custom_data": loads_or(row.get(VersionSchemaField.CUSTOM_DATA), None),
        },
    )


def read_tags_by_owner(payload_dir: str) -> Dict[tuple, List[dict]]:
    """Tag assignments grouped by what they hang on, in the API's own shape."""
    from supervisely.project.versioning.tag_schema import tag_json_from_row

    out: Dict[tuple, List[dict]] = {}
    for row in read_all_rows(payload_dir, TAGS_TABLE):
        key = (row[VersionSchemaField.OWNER_TYPE], row[VersionSchemaField.OWNER_ID])
        out.setdefault(key, []).append(tag_json_from_row(row))
    return out


def read_payload(payload_dir: str) -> Tuple[Any, Any, list, list, dict, dict]:
    """The unpacked snapshot as the six-part payload a restore works from.

    Same shape the pickle format carried - ``(project_info, meta, dataset_infos,
    image_infos, figures, alpha_geometries)`` - so both formats feed one restore.
    """
    from supervisely.api.project_api import ProjectInfo
    from supervisely.io.json import load_json_file
    from supervisely.project.project_meta import ProjectMeta

    project_info = namedtuple_from_dict(
        ProjectInfo, load_json_file(os.path.join(payload_dir, PROJECT_INFO_FILE))
    )
    meta = ProjectMeta.from_json(load_json_file(os.path.join(payload_dir, PROJECT_META_FILE)))

    dataset_infos = [build_dataset_info(row) for row in read_all_rows(payload_dir, DATASETS_TABLE)]

    tags_by_owner = read_tags_by_owner(payload_dir)
    image_infos = [
        build_image_info(row, tags=tags_by_owner.get(
            (OWNER_ITEM, row[VersionSchemaField.SRC_IMAGE_ID])))
        for row in read_all_rows(payload_dir, IMAGES_TABLE)
    ]

    alpha_geometries: Dict[int, dict] = {}
    for row in read_all_rows(payload_dir, ALPHA_GEOMETRIES_TABLE):
        alpha_geometries[row[VersionSchemaField.SRC_FIGURE_ID]] = loads_or(
            row.get(VersionSchemaField.GEOMETRY_JSON), None
        )

    figures: Dict[int, list] = {}
    for row in read_all_rows(payload_dir, FIGURES_TABLE):
        figure_id = row[VersionSchemaField.SRC_FIGURE_ID]
        figure = build_figure_info(
            row,
            alpha_geometry=alpha_geometries.get(figure_id),
            tags=tags_by_owner.get((OWNER_FIGURE, figure_id)),
        )
        figures.setdefault(row[VersionSchemaField.SRC_IMAGE_ID], []).append(figure)

    return project_info, meta, dataset_infos, image_infos, figures, alpha_geometries


class ImageSnapshotWriter:
    """Builds the Parquet payload of an image snapshot and packs it into a ``tar.zst``.

    Rows are handed over one at a time and flushed in row groups, so a snapshot can be
    written straight off an API stream without the project ever being in memory. Used
    both by :func:`Project.build_snapshot` (rows from the API) and by
    :func:`Project.repack_snapshot` (rows from an already-loaded pickle payload).
    """

    def __init__(self, schema_version: str, batch_rows: int = 5000):
        from supervisely.project.versioning.common import get_image_snapshot_schema
        from supervisely.project.versioning.tag_schema import get_tag_schema

        self.schema_version = schema_version
        self.schema = get_image_snapshot_schema(schema_version)
        self.tag_schema = get_tag_schema()
        self._batch_rows = batch_rows
        self._tmp_root: Optional[str] = None
        self._payload_dir: Optional[str] = None
        self._writers: Dict[str, Any] = {}

    def __enter__(self) -> "ImageSnapshotWriter":
        import tempfile

        from supervisely.io.fs import mkdir
        from supervisely.project.versioning.container import ParquetTableWriter

        pyarrow, parquet = import_pyarrow()

        self._tmp_root = tempfile.mkdtemp()
        self._payload_dir = os.path.join(self._tmp_root, "payload")
        mkdir(self._payload_dir)

        table_schemas = {
            DATASETS_TABLE: self.schema.datasets_schema(pyarrow),
            IMAGES_TABLE: self.schema.images_schema(pyarrow),
            FIGURES_TABLE: self.schema.figures_schema(pyarrow),
            ALPHA_GEOMETRIES_TABLE: self.schema.alpha_geometries_schema(pyarrow),
            TAGS_TABLE: self.tag_schema.tags_schema(pyarrow),
        }
        self._writers = {
            name: ParquetTableWriter(
                pyarrow,
                parquet,
                os.path.join(self._payload_dir, f"{name}.parquet"),
                table_schema,
                batch_rows=self._batch_rows,
            )
            for name, table_schema in table_schemas.items()
        }
        return self

    def __exit__(self, *exc_info) -> None:
        import shutil

        if self._tmp_root is not None:
            shutil.rmtree(self._tmp_root, ignore_errors=True)
            self._tmp_root = None

    def set_project(self, project_info, meta) -> None:
        from supervisely.io.json import dump_json_file

        dump_json_file(project_info._asdict(), os.path.join(self._payload_dir, PROJECT_INFO_FILE))
        dump_json_file(meta.to_json(), os.path.join(self._payload_dir, PROJECT_META_FILE))

    def add_dataset(self, ds_info, full_path: str) -> None:
        self._writers[DATASETS_TABLE].add(
            self.schema.dataset_row_from_ds_info(ds_info, full_path=full_path)
        )

    def add_image(self, image_info) -> None:
        self._writers[IMAGES_TABLE].add(self.schema.image_row_from_image_info(image_info))

    def add_figure(self, figure_info, *, src_image_id: int, class_name, store_geometry: bool):
        self._writers[FIGURES_TABLE].add(
            self.schema.figure_row_from_figure_info(
                figure_info,
                src_image_id=src_image_id,
                class_name=class_name,
                store_geometry=store_geometry,
            )
        )

    def add_tag(self, tag_json, *, owner_type: str, owner_id: int, src_item_id=None) -> None:
        self._writers[TAGS_TABLE].add(
            self.tag_schema.tag_row(
                tag_json, owner_type=owner_type, owner_id=owner_id, src_item_id=src_item_id
            )
        )

    def add_alpha_geometry(self, figure_id: int, geometry) -> None:
        self._writers[ALPHA_GEOMETRIES_TABLE].add(
            self.schema.alpha_geometry_row(src_figure_id=figure_id, geometry=geometry)
        )

    def finish(self):
        """Close the tables, write the manifest and return the packed snapshot."""
        from supervisely.io.json import dump_json_file
        from supervisely.project.versioning.container import pack_payload_dir, table_meta

        tables_meta = []
        for name, writer in self._writers.items():
            row_count = writer.close()
            if row_count > 0:
                tables_meta.append(table_meta(name, f"{name}.parquet", row_count))

        dump_json_file(
            {
                VersionSchemaField.SCHEMA_VERSION: self.schema_version,
                VersionSchemaField.TABLES: tables_meta,
            },
            os.path.join(self._payload_dir, "manifest.json"),
        )
        return pack_payload_dir(self._payload_dir, self._tmp_root)


def dataset_full_paths(dataset_infos) -> Dict[int, str]:
    """Map dataset id -> the path a snapshot stores, derived from the parent chain.

    ``api.dataset.tree()`` hands the parents over directly, but a payload loaded from a
    pickle backup only has ``parent_id``, so the chain has to be walked.
    """
    from supervisely.project.project import Dataset

    by_id = {ds.id: ds for ds in dataset_infos}

    def ancestor_names(ds) -> List[str]:
        names: List[str] = []
        seen = {ds.id}
        parent_id = getattr(ds, "parent_id", None)
        # A cycle cannot happen in a healthy project; refusing to loop forever on a
        # broken one is cheaper than finding out in production.
        while parent_id is not None and parent_id in by_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_id[parent_id]
            names.append(parent.name)
            parent_id = getattr(parent, "parent_id", None)
        names.reverse()
        return names

    return {ds.id: Dataset._get_dataset_path(ds.name, ancestor_names(ds)) for ds in dataset_infos}


def iter_arrow_batches(
    payload_dir: str,
    table: str,
    columns: Optional[Sequence[str]] = None,
    batch_size: int = 5000,
):
    """Yield ``pyarrow.RecordBatch`` from one snapshot table, without touching Python objects.

    The dict-per-row form the rest of this module produces is convenient and, at scale,
    is the whole cost: reading three columns of 4.5M figure rows takes 0.13 s as Arrow
    batches and 4.6 s once every row becomes a dict. A consumer that compares or hashes
    columns should stay in Arrow.
    """
    path = table_path(payload_dir, table)
    if path is None:
        return
    parquet = import_parquet()
    columns = existing_columns(path, columns)
    parquet_file = parquet.ParquetFile(path)
    try:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=list(columns or []) or None
        ):
            yield batch
    finally:
        parquet_file.close()
