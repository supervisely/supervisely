"""Read-only access to the contents of a project version, without restoring it.

A version snapshot already holds everything a comparison between two versions needs -
the meta, the dataset tree, the items and the figures. Restoring it into a project to
read it back costs a full upload; this reads the archive instead.

The interface is the same for every project type and both image schemas, and rows come
out under one set of column names (:class:`SnapshotColumn`) whatever the backend stores
them as. What is *not* uniform is the cost, and callers are entitled to know: a Parquet
snapshot streams in row groups, so peak memory is the batch size, while an image
snapshot in the legacy pickle format has to be loaded whole before the first row can be
yielded. :attr:`VersionSnapshot.is_columnar` says which one you have.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Union

from supervisely.api.module_api import ApiField
from supervisely.project.project_type import ProjectType
from supervisely.project.versioning import image_snapshot_io
from supervisely.project.versioning.common import IMAGE_SCHEMA_VERSION_V1
from supervisely.project.versioning.container import (
    SNIFF_SIZE,
    is_snapshot_container,
    read_manifest_schema_version,
    unpack_snapshot,
)
from supervisely.project.versioning.schema_fields import VersionSchemaField
from supervisely.sly_logger import logger


class SnapshotColumn:
    """Column names rows come out under, whatever the snapshot stores them as."""

    ITEM_ID = "item_id"
    DATASET_ID = "dataset_id"
    NAME = "name"
    HASH = "hash"
    LINK = "link"
    WIDTH = "width"
    HEIGHT = "height"
    FRAMES_COUNT = "frames_count"
    META = "meta"
    TAGS = "tags"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"

    # Tags are their own entity - see VersionSnapshot.iter_tags - so they are not a
    # column of the thing they hang on.
    TAG_ASSIGNMENT_ID = "tag_assignment_id"
    OWNER_TYPE = "owner_type"
    OWNER_ID = "owner_id"
    TAG_ID = "tag_id"
    VALUE = "value"
    FRAME_FROM = "frame_from"
    FRAME_TO = "frame_to"

    FIGURE_ID = "figure_id"
    CLASS_NAME = "class_name"
    GEOMETRY_TYPE = "geometry_type"
    GEOMETRY = "geometry"
    FRAME_INDEX = "frame_index"


ITEM_COLUMNS = (
    SnapshotColumn.ITEM_ID,
    SnapshotColumn.DATASET_ID,
    SnapshotColumn.NAME,
    SnapshotColumn.HASH,
    SnapshotColumn.LINK,
    SnapshotColumn.WIDTH,
    SnapshotColumn.HEIGHT,
    SnapshotColumn.FRAMES_COUNT,
    SnapshotColumn.META,
    SnapshotColumn.CREATED_AT,
    SnapshotColumn.UPDATED_AT,
)

FIGURE_COLUMNS = (
    SnapshotColumn.FIGURE_ID,
    SnapshotColumn.ITEM_ID,
    SnapshotColumn.CLASS_NAME,
    SnapshotColumn.GEOMETRY_TYPE,
    SnapshotColumn.GEOMETRY,
    SnapshotColumn.FRAME_INDEX,
    SnapshotColumn.META,
)

# Rows differ by three orders of magnitude in weight - an id and a name against a
# figure carrying a mask - so this is a safety default, not a throughput one. Measured
# on a 3000-image, 9000-figure mask-dense project, reading every column including
# geometry: 250 rows peaked at 91 MB, 1000 at 138 MB, 5000 at 260 MB. A caller reading
# a few columns can raise it freely.
DEFAULT_BATCH_SIZE = 1000


class SnapshotDataset(NamedTuple):
    """One dataset of a snapshot. The tree is small enough to always read whole."""

    id: int
    parent_id: Optional[int]
    name: str
    full_path: Optional[str]
    description: Optional[str] = None
    custom_data: Optional[dict] = None


def _select(row: Dict[str, Any], columns: Optional[Sequence[str]]) -> Dict[str, Any]:
    if columns is None:
        return row
    return {column: row.get(column) for column in columns}


class _Backend:
    """What every snapshot format has to provide. Subclasses are per format."""

    schema_version: str = ""
    is_columnar: bool = False
    # Whether FIGURE_ID and the object id behind CLASS_NAME are the server's own ids.
    # False means they are positions in the table, which shift when anything is inserted
    # or removed - so two versions cannot be matched on them.
    figure_ids_are_server_ids: bool = True

    @property
    def project_info(self):
        raise NotImplementedError

    @property
    def meta(self):
        raise NotImplementedError

    def datasets(self) -> List[SnapshotDataset]:
        raise NotImplementedError

    def iter_items(self, batch_size: int, columns: Optional[Sequence[str]]) -> Iterator[List[dict]]:
        raise NotImplementedError

    def iter_figures(
        self, batch_size: int, columns: Optional[Sequence[str]], with_geometry: bool
    ) -> Iterator[List[dict]]:
        raise NotImplementedError

    def iter_tags(self, batch_size: int, columns: Optional[Sequence[str]]):
        raise NotImplementedError(f"{type(self).__name__} does not expose tags")

    def iter_arrow(self, kind: str, batch_size: int, columns: Optional[Sequence[str]]):
        raise NotImplementedError(
            f"{type(self).__name__} cannot serve Arrow batches: its {kind} are not stored "
            "as columns. Use the dict interface."
        )


class _PayloadBackend(_Backend):
    """Shared plumbing for the Parquet formats: an unpacked payload directory."""

    def __init__(self, payload_dir: str, schema_version: str):
        from supervisely.io.json import load_json_file
        from supervisely.project.project_meta import ProjectMeta
        from supervisely.api.project_api import ProjectInfo

        self._payload_dir = payload_dir
        self.schema_version = schema_version
        self.is_columnar = True
        self._project_info = image_snapshot_io.namedtuple_from_dict(
            ProjectInfo,
            load_json_file(os.path.join(payload_dir, image_snapshot_io.PROJECT_INFO_FILE)),
        )
        self._meta = ProjectMeta.from_json(
            load_json_file(os.path.join(payload_dir, image_snapshot_io.PROJECT_META_FILE))
        )

    @property
    def project_info(self):
        return self._project_info

    @property
    def meta(self):
        return self._meta

    def iter_tags(self, batch_size: int, columns: Optional[Sequence[str]]):
        """Tag assignments, one row each, whatever they hang on."""
        for batch in image_snapshot_io.iter_rows(
            self._payload_dir, image_snapshot_io.TAGS_TABLE, batch_size=batch_size
        ):
            yield [_tag_row(row, columns) for row in batch]

    def _iter_table(
        self, table: str, physical_columns: Optional[Sequence[str]], batch_size: int
    ) -> Iterator[List[dict]]:
        return image_snapshot_io.iter_rows(
            self._payload_dir, table, columns=physical_columns, batch_size=batch_size
        )

    # kind -> (table name, canonical->stored mapping). A kind that is not here is not
    # column-shaped in this format and has no Arrow path.
    _ARROW_TABLES: Dict[str, tuple] = {}

    def iter_arrow(self, kind: str, batch_size: int, columns: Optional[Sequence[str]]):
        entry = self._ARROW_TABLES.get(kind)
        if entry is None:
            return super().iter_arrow(kind, batch_size, columns)
        table, mapping = entry

        if columns is not None:
            derived = [c for c in columns if mapping.get(c) is None]
            if derived:
                raise ValueError(
                    f"{', '.join(sorted(derived))} is not a stored column of this snapshot's "
                    f"{kind} - it is derived when rows are built. Read it through the dict "
                    "interface, or read the columns it is derived from."
                )

        physical = _physical_columns(mapping, columns)
        rename = {stored: canonical for canonical, stored in mapping.items() if stored}
        for batch in image_snapshot_io.iter_arrow_batches(
            self._payload_dir, table, columns=physical, batch_size=batch_size
        ):
            yield batch.rename_columns([rename.get(n, n) for n in batch.schema.names])


def _stored_tag(tag_json, owner_type, owner_id, item_id) -> Dict[str, Any]:
    """A tag from an annotation document in the same shape the tags table stores."""
    from supervisely.project.versioning.tag_schema import get_tag_schema

    return get_tag_schema().tag_row(
        tag_json, owner_type=owner_type, owner_id=owner_id, src_item_id=item_id
    )


def _tag_row(row: Dict[str, Any], columns: Optional[Sequence[str]]) -> Dict[str, Any]:
    """A stored tag row under canonical names, with the value decoded back to its type."""
    raw_value = row.get(VersionSchemaField.VALUE_JSON)
    return _select(
        {
            SnapshotColumn.TAG_ASSIGNMENT_ID: row.get(VersionSchemaField.TAG_ASSIGNMENT_ID),
            SnapshotColumn.OWNER_TYPE: row.get(VersionSchemaField.OWNER_TYPE),
            SnapshotColumn.OWNER_ID: row.get(VersionSchemaField.OWNER_ID),
            SnapshotColumn.ITEM_ID: row.get(VersionSchemaField.SRC_ITEM_ID),
            SnapshotColumn.TAG_ID: row.get(VersionSchemaField.TAG_ID),
            SnapshotColumn.NAME: row.get(VersionSchemaField.NAME),
            SnapshotColumn.VALUE: json.loads(raw_value) if raw_value is not None else None,
            SnapshotColumn.FRAME_FROM: row.get(VersionSchemaField.FRAME_FROM),
            SnapshotColumn.FRAME_TO: row.get(VersionSchemaField.FRAME_TO),
            SnapshotColumn.CREATED_AT: row.get(VersionSchemaField.CREATED_AT),
            SnapshotColumn.UPDATED_AT: row.get(VersionSchemaField.UPDATED_AT),
        },
        columns,
    )


def _wants_geometry(columns: Optional[Sequence[str]], with_geometry: bool) -> bool:
    """Whether geometry is worth reading at all.

    ``with_geometry`` defaults to True, so a caller that projects a few columns without
    naming GEOMETRY would otherwise pay for payloads that ``_select`` then throws away -
    and on the image backend that means walking the alpha-mask table for nothing.
    """
    return with_geometry and (columns is None or SnapshotColumn.GEOMETRY in columns)


def _physical_columns(mapping: Dict[str, str], columns: Optional[Sequence[str]]):
    """Stored columns needed to produce ``columns``.

    ``columns=None`` means every column the reader can surface - which is not the same
    as every column in the file. A video row stores the whole annotation as ``ann_json``
    beside the fields it exposes, and reading it to then ignore it cost 330 MB on a
    14k-video snapshot. Nothing outside this mapping is ever read.
    """
    wanted = mapping.keys() if columns is None else columns
    needed = {mapping[c] for c in wanted if mapping.get(c) is not None}
    return sorted(needed) or None


class _AlphaGeometryCursor:
    """Streams alpha-mask payloads alongside the figures table instead of preloading them.

    Alpha masks are the heaviest rows in an image snapshot - on a mask-dense project they
    are most of it - and both writers append an alpha row immediately after the figure it
    belongs to, so the two tables are in the same order. Walking them together keeps only
    one batch of masks alive; reading the alpha table into a dict first would put every
    mask in the project in memory, which is exactly what the format is meant to avoid.

    The order is an invariant of how snapshots are written, not something a reader can
    assume of a file it did not produce, so a row arrived at out of turn is stashed rather
    than dropped. In the ordered case ``_stash`` stays empty.
    """

    def __init__(self, payload_dir: str, batch_size: int):
        self._batches = image_snapshot_io.iter_rows(
            payload_dir, image_snapshot_io.ALPHA_GEOMETRIES_TABLE, batch_size=batch_size
        )
        self._current: List[dict] = []
        self._pos = 0
        self._stash: Dict[int, Any] = {}
        self._exhausted = False

    def _next_row(self) -> Optional[dict]:
        while self._pos >= len(self._current):
            if self._exhausted:
                return None
            try:
                self._current = next(self._batches)
            except StopIteration:
                self._exhausted = True
                return None
            self._pos = 0
        row = self._current[self._pos]
        self._pos += 1
        return row

    def get(self, figure_id: Optional[int]):
        if figure_id is None:
            return None
        if figure_id in self._stash:
            return self._stash.pop(figure_id)
        while True:
            row = self._next_row()
            if row is None:
                return None
            geometry = image_snapshot_io.loads_or(row.get(VersionSchemaField.GEOMETRY_JSON), None)
            row_id = row.get(VersionSchemaField.SRC_FIGURE_ID)
            if row_id == figure_id:
                return geometry
            self._stash[row_id] = geometry


class _ImagesV2Backend(_PayloadBackend):
    """Images, Parquet. The one format where every canonical column is a stored column."""

    _ITEM_MAP = {
        SnapshotColumn.ITEM_ID: VersionSchemaField.SRC_IMAGE_ID,
        SnapshotColumn.DATASET_ID: VersionSchemaField.SRC_DATASET_ID,
        SnapshotColumn.NAME: VersionSchemaField.NAME,
        SnapshotColumn.HASH: VersionSchemaField.HASH,
        SnapshotColumn.LINK: VersionSchemaField.LINK,
        SnapshotColumn.WIDTH: VersionSchemaField.WIDTH,
        SnapshotColumn.HEIGHT: VersionSchemaField.HEIGHT,
        SnapshotColumn.FRAMES_COUNT: None,
        SnapshotColumn.META: VersionSchemaField.META,
        SnapshotColumn.CREATED_AT: VersionSchemaField.CREATED_AT,
        SnapshotColumn.UPDATED_AT: VersionSchemaField.UPDATED_AT,
    }

    _FIGURE_MAP = {
        SnapshotColumn.FIGURE_ID: VersionSchemaField.SRC_FIGURE_ID,
        SnapshotColumn.ITEM_ID: VersionSchemaField.SRC_IMAGE_ID,
        SnapshotColumn.CLASS_NAME: VersionSchemaField.CLASS_NAME,
        SnapshotColumn.GEOMETRY_TYPE: VersionSchemaField.GEOMETRY_TYPE,
        SnapshotColumn.GEOMETRY: VersionSchemaField.GEOMETRY_JSON,
        SnapshotColumn.FRAME_INDEX: None,
        SnapshotColumn.META: VersionSchemaField.META,
        SnapshotColumn.CREATED_AT: VersionSchemaField.CREATED_AT,
        SnapshotColumn.UPDATED_AT: VersionSchemaField.UPDATED_AT,
    }

    @property
    def _ARROW_TABLES(self):
        # GEOMETRY on this path is the stored column: an alpha mask's payload lives in
        # its own table and is joined only when rows are built.
        return {
            "items": (image_snapshot_io.IMAGES_TABLE, self._ITEM_MAP),
            "figures": (image_snapshot_io.FIGURES_TABLE, self._FIGURE_MAP),
        }

    def datasets(self) -> List[SnapshotDataset]:
        return [
            SnapshotDataset(
                id=row[VersionSchemaField.SRC_DATASET_ID],
                parent_id=row.get(VersionSchemaField.PARENT_SRC_DATASET_ID),
                name=row[VersionSchemaField.NAME],
                full_path=row.get(VersionSchemaField.FULL_PATH),
                description=row.get(VersionSchemaField.DESCRIPTION),
                custom_data=image_snapshot_io.loads_or(
                    row.get(VersionSchemaField.CUSTOM_DATA), {}
                ),
            )
            for row in image_snapshot_io.read_all_rows(
                self._payload_dir, image_snapshot_io.DATASETS_TABLE
            )
        ]

    def iter_items(self, batch_size, columns):
        physical = _physical_columns(self._ITEM_MAP, columns)
        for batch in self._iter_table(image_snapshot_io.IMAGES_TABLE, physical, batch_size):
            yield [
                _select(
                    {
                        SnapshotColumn.ITEM_ID: row.get(VersionSchemaField.SRC_IMAGE_ID),
                        SnapshotColumn.DATASET_ID: row.get(VersionSchemaField.SRC_DATASET_ID),
                        SnapshotColumn.NAME: row.get(VersionSchemaField.NAME),
                        SnapshotColumn.HASH: row.get(VersionSchemaField.HASH),
                        SnapshotColumn.LINK: row.get(VersionSchemaField.LINK),
                        SnapshotColumn.WIDTH: row.get(VersionSchemaField.WIDTH),
                        SnapshotColumn.HEIGHT: row.get(VersionSchemaField.HEIGHT),
                        SnapshotColumn.FRAMES_COUNT: None,
                        SnapshotColumn.META: image_snapshot_io.loads_or(
                            row.get(VersionSchemaField.META), {}
                        ),
                        SnapshotColumn.CREATED_AT: row.get(VersionSchemaField.CREATED_AT),
                        SnapshotColumn.UPDATED_AT: row.get(VersionSchemaField.UPDATED_AT),
                    },
                    columns,
                )
                for row in batch
            ]

    def iter_figures(self, batch_size, columns, with_geometry):
        from supervisely.geometry.alpha_mask import AlphaMask

        wants_geometry = _wants_geometry(columns, with_geometry)
        physical = _physical_columns(self._FIGURE_MAP, columns)
        if wants_geometry and physical is not None:
            # The type decides whether a null geometry means "inline, absent" or "in the
            # alpha table", so it has to be read even when the caller did not ask for it.
            physical = sorted(
                set(physical)
                | {VersionSchemaField.GEOMETRY_JSON, VersionSchemaField.GEOMETRY_TYPE}
            )
        alpha = _AlphaGeometryCursor(self._payload_dir, batch_size) if wants_geometry else None
        alpha_mask_name = AlphaMask.name()
        for batch in self._iter_table(image_snapshot_io.FIGURES_TABLE, physical, batch_size):
            rows = []
            for row in batch:
                geometry = None
                if wants_geometry:
                    geometry = image_snapshot_io.loads_or(
                        row.get(VersionSchemaField.GEOMETRY_JSON), None
                    )
                    if geometry is None and row.get(VersionSchemaField.GEOMETRY_TYPE) == (
                        alpha_mask_name
                    ):
                        geometry = alpha.get(row.get(VersionSchemaField.SRC_FIGURE_ID))
                rows.append(
                    _select(
                        {
                            SnapshotColumn.FIGURE_ID: row.get(VersionSchemaField.SRC_FIGURE_ID),
                            SnapshotColumn.ITEM_ID: row.get(VersionSchemaField.SRC_IMAGE_ID),
                            SnapshotColumn.CLASS_NAME: row.get(VersionSchemaField.CLASS_NAME),
                            SnapshotColumn.GEOMETRY_TYPE: row.get(
                                VersionSchemaField.GEOMETRY_TYPE
                            ),
                            SnapshotColumn.GEOMETRY: geometry,
                            SnapshotColumn.FRAME_INDEX: None,
                            SnapshotColumn.META: image_snapshot_io.loads_or(
                                row.get(VersionSchemaField.META), {}
                            ),
                            SnapshotColumn.CREATED_AT: row.get(VersionSchemaField.CREATED_AT),
                            SnapshotColumn.UPDATED_AT: row.get(VersionSchemaField.UPDATED_AT),
                        },
                        columns,
                    )
                )
            yield rows



class _ImagesV1Backend(_Backend):
    """Images, pickle.

    The payload has to be loaded whole - that is the format, and it is why
    :func:`Project.repack_snapshot` exists. Rows are handed out in batches anyway so
    callers can be written once against the streaming interface.
    """

    schema_version = IMAGE_SCHEMA_VERSION_V1
    is_columnar = False

    def __init__(self, payload):
        (
            self._project_info,
            self._meta,
            self._dataset_infos,
            self._image_infos,
            self._figures,
            self._alpha_geometries,
        ) = payload
        self._class_name_by_id = {
            obj_class.sly_id: obj_class.name
            for obj_class in self._meta.obj_classes
            if obj_class.sly_id is not None
        }

    @property
    def project_info(self):
        return self._project_info

    @property
    def meta(self):
        return self._meta

    def datasets(self) -> List[SnapshotDataset]:
        full_paths = image_snapshot_io.dataset_full_paths(self._dataset_infos)
        return [
            SnapshotDataset(
                id=ds.id,
                parent_id=getattr(ds, "parent_id", None),
                name=ds.name,
                full_path=full_paths.get(ds.id),
                description=getattr(ds, "description", None),
                custom_data=getattr(ds, "custom_data", None) or {},
            )
            for ds in self._dataset_infos
        ]

    def iter_items(self, batch_size, columns):
        batch = []
        for image_info in self._image_infos:
            batch.append(
                _select(
                    {
                        SnapshotColumn.ITEM_ID: image_info.id,
                        SnapshotColumn.DATASET_ID: image_info.dataset_id,
                        SnapshotColumn.NAME: image_info.name,
                        SnapshotColumn.HASH: getattr(image_info, "hash", None),
                        SnapshotColumn.LINK: getattr(image_info, "link", None),
                        SnapshotColumn.WIDTH: getattr(image_info, "width", None),
                        SnapshotColumn.HEIGHT: getattr(image_info, "height", None),
                        SnapshotColumn.FRAMES_COUNT: None,
                        SnapshotColumn.META: getattr(image_info, "meta", None) or {},
                        SnapshotColumn.CREATED_AT: getattr(image_info, "created_at", None),
                        SnapshotColumn.UPDATED_AT: getattr(image_info, "updated_at", None),
                    },
                    columns,
                )
            )
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def iter_tags(self, batch_size, columns):
        """A pickle keeps tags on the objects they hang on; the stream is the same."""
        from supervisely.project.versioning.tag_schema import OWNER_FIGURE, OWNER_ITEM

        batch = []
        for image_info in self._image_infos:
            for tag_json in getattr(image_info, "tags", None) or []:
                batch.append(
                    _tag_row(
                        _stored_tag(tag_json, OWNER_ITEM, image_info.id, image_info.id), columns
                    )
                )
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        for image_id, image_figures in self._figures.items():
            for figure in image_figures:
                for tag_json in getattr(figure, "tags", None) or []:
                    batch.append(
                        _tag_row(
                            _stored_tag(tag_json, OWNER_FIGURE, figure.id, image_id), columns
                        )
                    )
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
        if batch:
            yield batch

    def iter_figures(self, batch_size, columns, with_geometry):
        batch = []
        for image_id, image_figures in self._figures.items():
            for figure in image_figures:
                geometry = None
                if with_geometry:
                    geometry = getattr(figure, "geometry", None)
                    if geometry is None:
                        geometry = self._alpha_geometries.get(figure.id)
                batch.append(
                    _select(
                        {
                            SnapshotColumn.FIGURE_ID: figure.id,
                            SnapshotColumn.ITEM_ID: image_id,
                            SnapshotColumn.CLASS_NAME: self._class_name_by_id.get(
                                getattr(figure, "class_id", None)
                            ),
                            SnapshotColumn.GEOMETRY_TYPE: getattr(figure, "geometry_type", None),
                            SnapshotColumn.GEOMETRY: geometry,
                            SnapshotColumn.FRAME_INDEX: None,
                            SnapshotColumn.META: getattr(figure, "meta", None) or {},
                            SnapshotColumn.CREATED_AT: getattr(figure, "created_at", None),
                            SnapshotColumn.UPDATED_AT: getattr(figure, "updated_at", None),
                        },
                        columns,
                    )
                )
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch


class _VideoV2Backend(_PayloadBackend):
    """Videos, Parquet. Figures are normalized into their own table already."""

    @property
    def figure_ids_are_server_ids(self) -> bool:
        from supervisely.project.versioning.common import VIDEO_SCHEMA_VERSION_V2

        # v2.0.0 numbered objects and figures by table position and kept the real ids
        # only in key_id_map.json, under a uuid the SDK invented at snapshot time.
        return self.schema_version != VIDEO_SCHEMA_VERSION_V2

    _ITEM_MAP = {
        SnapshotColumn.ITEM_ID: VersionSchemaField.SRC_VIDEO_ID,
        SnapshotColumn.DATASET_ID: VersionSchemaField.SRC_DATASET_ID,
        SnapshotColumn.NAME: VersionSchemaField.NAME,
        SnapshotColumn.HASH: VersionSchemaField.HASH,
        SnapshotColumn.LINK: VersionSchemaField.LINK,
        SnapshotColumn.WIDTH: VersionSchemaField.FRAME_WIDTH,
        SnapshotColumn.HEIGHT: VersionSchemaField.FRAME_HEIGHT,
        SnapshotColumn.FRAMES_COUNT: VersionSchemaField.FRAMES_COUNT,
        SnapshotColumn.META: VersionSchemaField.META,
        # A column from schema v2.1.0: the video's own tags, with their frame ranges,
        # which v2.0.0 kept only inside ann_json.
        SnapshotColumn.CREATED_AT: VersionSchemaField.CREATED_AT,
        SnapshotColumn.UPDATED_AT: VersionSchemaField.UPDATED_AT,
    }

    @property
    def _ARROW_TABLES(self):
        return {
            "items": ("videos", self._ITEM_MAP),
            "figures": ("figures", self._FIGURE_MAP),
        }

    def datasets(self) -> List[SnapshotDataset]:
        return [
            SnapshotDataset(
                id=row[VersionSchemaField.SRC_DATASET_ID],
                parent_id=row.get(VersionSchemaField.PARENT_SRC_DATASET_ID),
                name=row[VersionSchemaField.NAME],
                full_path=row.get(VersionSchemaField.FULL_PATH),
                description=row.get(VersionSchemaField.DESCRIPTION),
                custom_data=image_snapshot_io.loads_or(
                    row.get(VersionSchemaField.CUSTOM_DATA), {}
                ),
            )
            for row in image_snapshot_io.read_all_rows(
                self._payload_dir, image_snapshot_io.DATASETS_TABLE
            )
        ]

    def iter_items(self, batch_size, columns):
        physical = _physical_columns(self._ITEM_MAP, columns)
        for batch in self._iter_table("videos", physical, batch_size):
            yield [
                _select(
                    {
                        SnapshotColumn.ITEM_ID: row.get(VersionSchemaField.SRC_VIDEO_ID),
                        SnapshotColumn.DATASET_ID: row.get(VersionSchemaField.SRC_DATASET_ID),
                        SnapshotColumn.NAME: row.get(VersionSchemaField.NAME),
                        SnapshotColumn.HASH: row.get(VersionSchemaField.HASH),
                        SnapshotColumn.LINK: row.get(VersionSchemaField.LINK),
                        SnapshotColumn.WIDTH: row.get(VersionSchemaField.FRAME_WIDTH),
                        SnapshotColumn.HEIGHT: row.get(VersionSchemaField.FRAME_HEIGHT),
                        SnapshotColumn.FRAMES_COUNT: row.get(VersionSchemaField.FRAMES_COUNT),
                        SnapshotColumn.META: image_snapshot_io.loads_or(
                            row.get(VersionSchemaField.META), {}
                        ),
                        SnapshotColumn.CREATED_AT: row.get(VersionSchemaField.CREATED_AT),
                        SnapshotColumn.UPDATED_AT: row.get(VersionSchemaField.UPDATED_AT),
                    },
                    columns,
                )
                for row in batch
            ]

    _FIGURE_MAP = {
        SnapshotColumn.FIGURE_ID: VersionSchemaField.SRC_FIGURE_ID,
        SnapshotColumn.ITEM_ID: VersionSchemaField.SRC_VIDEO_ID,
        # Not on the figure row - joined from the objects table below.
        SnapshotColumn.CLASS_NAME: None,
        SnapshotColumn.GEOMETRY_TYPE: VersionSchemaField.GEOMETRY_TYPE,
        SnapshotColumn.GEOMETRY: VersionSchemaField.GEOMETRY_JSON,
        SnapshotColumn.FRAME_INDEX: VersionSchemaField.FRAME_INDEX,
        SnapshotColumn.META: None,
        # Present from schema v2.1.0; a v2.0.0 snapshot has no column for them.
        SnapshotColumn.CREATED_AT: VersionSchemaField.CREATED_AT,
        SnapshotColumn.UPDATED_AT: VersionSchemaField.UPDATED_AT,
    }

    def iter_figures(self, batch_size, columns, with_geometry):
        wants_geometry = _wants_geometry(columns, with_geometry)
        wants_object = columns is None or bool(
            {SnapshotColumn.CLASS_NAME} & set(columns)
        )

        physical = _physical_columns(self._FIGURE_MAP, columns)
        if physical is not None:
            if not wants_geometry:
                physical = [c for c in physical if c != VersionSchemaField.GEOMETRY_JSON]
            if wants_object:
                # The join key, needed whenever the class or the tags are.
                physical = sorted(set(physical) | {VersionSchemaField.SRC_OBJECT_ID})

        # One entry per annotation object, not per figure - orders of magnitude smaller
        # than the figures table, which is what makes indexing it in memory acceptable.
        # Skipped entirely when nothing that comes from it was asked for.
        objects = {}
        if wants_object:
            objects = {
                row[VersionSchemaField.SRC_OBJECT_ID]: row.get(VersionSchemaField.CLASS_NAME)
                for row in image_snapshot_io.read_all_rows(
                    self._payload_dir,
                    "objects",
                    columns=[
                        VersionSchemaField.SRC_OBJECT_ID,
                        VersionSchemaField.CLASS_NAME,
                    ],
                )
            }

        for batch in self._iter_table("figures", physical, batch_size):
            rows = []
            for row in batch:
                class_name = objects.get(row.get(VersionSchemaField.SRC_OBJECT_ID))
                rows.append(
                    _select(
                        {
                            SnapshotColumn.FIGURE_ID: row.get(VersionSchemaField.SRC_FIGURE_ID),
                            SnapshotColumn.ITEM_ID: row.get(VersionSchemaField.SRC_VIDEO_ID),
                            SnapshotColumn.CLASS_NAME: class_name,
                            SnapshotColumn.GEOMETRY_TYPE: row.get(
                                VersionSchemaField.GEOMETRY_TYPE
                            ),
                            SnapshotColumn.GEOMETRY: (
                                image_snapshot_io.loads_or(
                                    row.get(VersionSchemaField.GEOMETRY_JSON), None
                                )
                                if wants_geometry
                                else None
                            ),
                            SnapshotColumn.FRAME_INDEX: row.get(VersionSchemaField.FRAME_INDEX),
                            SnapshotColumn.META: {},
                            SnapshotColumn.CREATED_AT: row.get(VersionSchemaField.CREATED_AT),
                            SnapshotColumn.UPDATED_AT: row.get(VersionSchemaField.UPDATED_AT),
                        },
                        columns,
                    )
                )
            yield rows


class _VolumeSectionsBackend(_Backend):
    """Volumes.

    Unlike the other two this format is not a tar of Parquet files: it is one bespoke
    binary blob (magic ``SLYVOLPAR``) holding the project info, the meta and three
    Parquet tables as length-prefixed sections. The tables themselves are read the same
    way, straight out of the buffer.

    It also stores whole JSON records rather than columns - one per dataset, one per
    volume, one annotation per volume - so reading is parse-bound rather than
    column-bound, and there is no Arrow path to offer.
    """

    def __init__(self, blob: bytes):
        from supervisely.project.project_meta import ProjectMeta
        from supervisely.project.volume_project import VolumeProject
        from supervisely.api.project_api import ProjectInfo

        self.is_columnar = True
        sections = VolumeProject._parse_parquet_sections(blob)
        self._sections = sections
        # Header version 1 stored annotations with uuid keys the SDK invented per parse,
        # so nothing in them survives from one version to the next. Version 2 keeps the
        # server's ids alongside.
        header_version = sections.get(VolumeProject._SECTION_HEADER_VERSION, 1)
        self.figure_ids_are_server_ids = header_version >= 2
        self.schema_version = "v2.1.0" if header_version >= 2 else "v2.0.0"
        self._project_info = image_snapshot_io.namedtuple_from_dict(
            ProjectInfo,
            json.loads(sections[VolumeProject._SECTION_PROJECT_INFO].decode("utf-8")),
        )
        self._meta = ProjectMeta.from_json(
            json.loads(sections[VolumeProject._SECTION_PROJECT_META].decode("utf-8"))
        )
        self._datasets_section = VolumeProject._SECTION_DATASETS
        self._volumes_section = VolumeProject._SECTION_VOLUMES
        self._annotations_section = VolumeProject._SECTION_ANNOTATIONS

    @property
    def project_info(self):
        return self._project_info

    @property
    def meta(self):
        return self._meta

    def _iter_section(self, section: int, batch_size: int) -> Iterator[List[dict]]:
        blob = self._sections.get(section)
        if not blob:
            return
        import pyarrow

        for batch in image_snapshot_io.iter_rows_from_source(
            pyarrow.BufferReader(blob), batch_size=batch_size
        ):
            yield batch

    def datasets(self) -> List[SnapshotDataset]:
        datasets = []
        for batch in self._iter_section(self._datasets_section, 1000):
            for row in batch:
                record = image_snapshot_io.loads_or(row.get(VersionSchemaField.JSON), {})
                datasets.append(
                    SnapshotDataset(
                        id=row[VersionSchemaField.SRC_DATASET_ID],
                        parent_id=record.get(ApiField.PARENT_ID),
                        name=record.get(ApiField.NAME),
                        full_path=None,
                        description=record.get(ApiField.DESCRIPTION),
                        custom_data=record.get(ApiField.CUSTOM_DATA) or {},
                    )
                )
        return datasets

    def iter_items(self, batch_size, columns):
        for batch in self._iter_section(self._volumes_section, batch_size):
            rows = []
            for row in batch:
                record = image_snapshot_io.loads_or(row.get(VersionSchemaField.JSON), {})
                rows.append(
                    _select(
                        {
                            SnapshotColumn.ITEM_ID: row.get(VersionSchemaField.SRC_VOLUME_ID),
                            SnapshotColumn.DATASET_ID: row.get(VersionSchemaField.SRC_DATASET_ID),
                            SnapshotColumn.NAME: record.get(ApiField.NAME),
                            SnapshotColumn.HASH: record.get(ApiField.HASH),
                            SnapshotColumn.LINK: record.get(ApiField.LINK),
                            SnapshotColumn.WIDTH: None,
                            SnapshotColumn.HEIGHT: None,
                            SnapshotColumn.FRAMES_COUNT: None,
                            SnapshotColumn.META: record.get(ApiField.META) or {},
                            SnapshotColumn.CREATED_AT: record.get(ApiField.CREATED_AT),
                            SnapshotColumn.UPDATED_AT: record.get(ApiField.UPDATED_AT),
                        },
                        columns,
                    )
                )
            yield rows

    def iter_annotations(self, batch_size: int = 50) -> Iterator[List[dict]]:
        """``(volume_id, annotation_json)`` pairs, for callers that want the raw form."""
        for batch in self._iter_section(self._annotations_section, batch_size):
            yield [
                {
                    SnapshotColumn.ITEM_ID: row.get(VersionSchemaField.SRC_VOLUME_ID),
                    "annotation": image_snapshot_io.loads_or(
                        row.get(VersionSchemaField.ANNOTATION), {}
                    ),
                }
                for row in batch
            ]

    def iter_tags(self, batch_size, columns):
        """Volume tags live inside the annotation document, not in a table of their own -
        this format stores whole records - so they are flattened out of it here to keep
        the interface the same as the other backends."""
        from supervisely.volume_annotation import constants as volume_constants
        from supervisely.project.versioning.tag_schema import OWNER_ITEM, OWNER_OBJECT

        for batch in self.iter_annotations(batch_size=max(1, batch_size // 100)):
            rows = []
            for entry in batch:
                volume_id = entry[SnapshotColumn.ITEM_ID]
                annotation = entry["annotation"]
                for tag in annotation.get(volume_constants.TAGS, []) or []:
                    rows.append(
                        _tag_row(_stored_tag(tag, OWNER_ITEM, volume_id, volume_id), columns)
                    )
                for obj in annotation.get(volume_constants.OBJECTS, []):
                    for tag in obj.get(volume_constants.TAGS, []) or []:
                        rows.append(
                            _tag_row(
                                _stored_tag(
                                    tag, OWNER_OBJECT, obj.get(volume_constants.KEY), volume_id
                                ),
                                columns,
                            )
                        )
            yield rows

    def iter_figures(self, batch_size, columns, with_geometry):
        from supervisely.annotation.label import LabelJsonFields
        from supervisely.volume_annotation import constants as volume_constants

        wants_geometry = _wants_geometry(columns, with_geometry)

        for batch in self.iter_annotations(batch_size=max(1, batch_size // 100)):
            rows = []
            for entry in batch:
                volume_id = entry[SnapshotColumn.ITEM_ID]
                annotation = entry["annotation"]
                objects = annotation.get(volume_constants.OBJECTS, [])
                class_by_object_key = {
                    obj.get(volume_constants.KEY): obj.get(LabelJsonFields.OBJ_CLASS_NAME)
                    for obj in objects
                }
                def emit(figure, frame_index):
                    object_key = figure.get(volume_constants.OBJECT_KEY)
                    rows.append(
                        _select(
                            {
                                # The id when the snapshot has one, the uuid key otherwise.
                                SnapshotColumn.FIGURE_ID: figure.get(volume_constants.ID)
                                or figure.get(volume_constants.KEY),
                                SnapshotColumn.ITEM_ID: volume_id,
                                SnapshotColumn.CLASS_NAME: class_by_object_key.get(object_key),
                                SnapshotColumn.GEOMETRY_TYPE: figure.get(ApiField.GEOMETRY_TYPE),
                                SnapshotColumn.GEOMETRY: (
                                    figure.get(ApiField.GEOMETRY) if wants_geometry else None
                                ),
                                SnapshotColumn.FRAME_INDEX: frame_index,
                                SnapshotColumn.META: figure.get(volume_constants.META) or {},
                            },
                            columns,
                        )
                    )

                for plane in annotation.get(volume_constants.PLANES, []):
                    for volume_slice in plane.get(volume_constants.SLICES, []):
                        for figure in volume_slice.get(volume_constants.FIGURES, []):
                            emit(figure, volume_slice.get(volume_constants.INDEX))
                for figure in annotation.get(volume_constants.SPATIAL_FIGURES, []):
                    emit(figure, None)
            yield rows


class VersionSnapshot:
    """A project version, opened for reading.

    :Usage Example:

        .. code-block:: python

            with VersionSnapshot.open(api, project_id=17, version_id=42) as snapshot:
                print(snapshot.project_type, len(snapshot.datasets()))
                for batch in snapshot.iter_items(
                    columns=[SnapshotColumn.ITEM_ID, SnapshotColumn.NAME, SnapshotColumn.HASH]
                ):
                    ...
    """

    def __init__(self, backend: _Backend, cleanup_dirs: Optional[List[str]] = None):
        self._backend = backend
        self._cleanup_dirs = cleanup_dirs or []

    # --------------------------------------------------------------- opening

    @classmethod
    def open(
        cls,
        api,
        project: Union[int, Any],
        version_id: int,
        cache_dir: Optional[str] = None,
        repack_legacy: bool = False,
    ) -> "VersionSnapshot":
        """
        Download a version's snapshot and open it.

        ``cache_dir`` decides what happens to the downloaded archive. Without one, the
        snapshot goes to a temporary directory that :func:`close` removes. With one, it
        is kept under ``<cache_dir>/<project_id>/<version_id>/``, and a later open of the
        same version costs nothing - which is what makes comparing v4 to v7 and then v7
        to v8 download three snapshots rather than four. Nothing prunes that directory;
        the caller owns its lifetime.

        :param api: Supervisely API client.
        :param project: Project ID or ProjectInfo of the version's project.
        :param version_id: Version ID.
        :param cache_dir: Directory to keep downloaded snapshots in. None means a temporary one.
        :param repack_legacy: Convert an image snapshot in the legacy pickle format into the
            Parquet container once and read that from then on. Needs ``cache_dir`` to have
            anywhere to put it, and is off by default because the conversion has to load the
            pickle whole - it pays for itself from the second read of that version onwards.
        :returns: An open snapshot.
        :rtype: :class:`VersionSnapshot`
        """
        project_id = project if isinstance(project, int) else project.id

        cleanup_dirs = []
        if cache_dir is None:
            download_dir = tempfile.mkdtemp()
            cleanup_dirs.append(download_dir)
        else:
            download_dir = os.path.join(cache_dir, str(project_id), str(version_id))
            os.makedirs(download_dir, exist_ok=True)

        snapshot_path = os.path.join(download_dir, "version.bin")
        repacked_path = os.path.join(download_dir, "version.v2.bin")

        if cache_dir is not None and os.path.isfile(repacked_path):
            logger.debug(f"Reusing converted snapshot of version {version_id}: {repacked_path}")
            return cls.open_archive(
                repacked_path, payload_dir=os.path.join(download_dir, "payload")
            )

        if not os.path.isfile(snapshot_path):
            api.project.version.download_snapshot(project, version_id, dest_path=snapshot_path)
        else:
            logger.debug(f"Reusing cached snapshot of version {version_id}: {snapshot_path}")

        if repack_legacy and cache_dir is not None:
            snapshot_path = cls._repack_into_cache(snapshot_path, repacked_path, version_id)

        try:
            return cls.open_archive(
                snapshot_path,
                payload_dir=os.path.join(download_dir, "payload"),
                cleanup_dirs=cleanup_dirs,
            )
        except Exception:
            for directory in cleanup_dirs:
                shutil.rmtree(directory, ignore_errors=True)
            raise

    @staticmethod
    def _repack_into_cache(snapshot_path: str, repacked_path: str, version_id: int) -> str:
        """Convert a legacy pickle snapshot into the cache; return what to read.

        Flipping the writer does nothing for versions that already exist, so an old
        version keeps costing a whole-payload load on every read until it is converted.
        A failure here is not fatal - the pickle is still readable, just slowly.
        """
        with open(snapshot_path, "rb") as f:
            if is_snapshot_container(f.read(SNIFF_SIZE)):
                return snapshot_path

        from supervisely.project.project import Project

        try:
            converted = Project.repack_snapshot(snapshot_path)
        except Exception as e:
            logger.warning(
                f"Could not convert version {version_id} to the Parquet snapshot format, "
                f"reading it as a pickle instead: {e}"
            )
            return snapshot_path

        # Written aside and moved, so an interrupted conversion cannot leave a partial
        # file that later opens take for a finished one.
        partial_path = repacked_path + ".partial"
        with open(partial_path, "wb") as f:
            f.write(converted.getvalue())
        os.replace(partial_path, repacked_path)
        logger.info(f"Converted version {version_id} snapshot to Parquet: {repacked_path}")
        return repacked_path

    @classmethod
    def open_archive(
        cls,
        path: str,
        payload_dir: Optional[str] = None,
        cleanup_dirs: Optional[List[str]] = None,
    ) -> "VersionSnapshot":
        """
        Open a snapshot that is already on disk (the ``version.bin`` of a version archive).

        :param path: Path to the snapshot.
        :param payload_dir: Where to unpack a Parquet snapshot. A temporary directory by default.
        :param cleanup_dirs: Directories :func:`close` should remove.
        :returns: An open snapshot.
        :rtype: :class:`VersionSnapshot`
        """
        cleanup_dirs = list(cleanup_dirs or [])
        with open(path, "rb") as f:
            head = f.read(SNIFF_SIZE)

        from supervisely.project.volume_project import VolumeProject

        if head.startswith(VolumeProject._SERIALIZATION_MAGIC):
            # Volume snapshots are one self-describing blob, not a tar of files.
            with open(path, "rb") as f:
                return cls(_VolumeSectionsBackend(f.read()), cleanup_dirs)

        if not is_snapshot_container(head):
            from supervisely.project.project import CustomUnpickler

            with open(path, "rb") as f:
                return cls(_ImagesV1Backend(CustomUnpickler(f).load()), cleanup_dirs)

        if payload_dir is None:
            payload_dir = tempfile.mkdtemp()
            cleanup_dirs.append(payload_dir)

        # An already-unpacked payload is reused, so a cached snapshot is not re-extracted
        # on every open either.
        if not os.path.isfile(os.path.join(payload_dir, "manifest.json")):
            os.makedirs(payload_dir, exist_ok=True)
            with open(path, "rb") as f:
                unpack_snapshot(f.read(), payload_dir)

        schema_version = read_manifest_schema_version(payload_dir)
        backend = cls._backend_for(payload_dir, schema_version)
        return cls(backend, cleanup_dirs)

    @staticmethod
    def _backend_for(payload_dir: str, schema_version: Optional[str]) -> _Backend:
        """Pick the backend from the project type recorded in the payload."""
        from supervisely.io.json import load_json_file

        project_info = load_json_file(
            os.path.join(payload_dir, image_snapshot_io.PROJECT_INFO_FILE)
        )
        project_type = project_info.get("type")

        # Each format validates its own version, so a snapshot written by a newer SDK is
        # refused rather than read with whichever columns happen to line up.
        from supervisely.project.versioning.common import (
            get_image_snapshot_schema,
            get_video_snapshot_schema,
        )

        if project_type == ProjectType.IMAGES.value:
            get_image_snapshot_schema(schema_version)
            return _ImagesV2Backend(payload_dir, schema_version)
        if project_type == ProjectType.VIDEOS.value:
            get_video_snapshot_schema(schema_version)
            return _VideoV2Backend(payload_dir, schema_version)
        if project_type == ProjectType.VOLUMES.value:
            raise RuntimeError(
                "A volume snapshot is not a tar of Parquet files - it is one blob with "
                "its own header, and is opened by that instead."
            )
        raise RuntimeError(f"Snapshots are not supported for project type {project_type!r}")

    # --------------------------------------------------------------- reading

    @property
    def project_info(self):
        return self._backend.project_info

    @property
    def meta(self):
        """The project's :class:`~supervisely.project.project_meta.ProjectMeta` at that version."""
        return self._backend.meta

    @property
    def project_type(self) -> Optional[str]:
        return getattr(self._backend.project_info, "type", None)

    @property
    def schema_version(self) -> str:
        return self._backend.schema_version

    @property
    def is_columnar(self) -> bool:
        """False for the legacy pickle format, whose payload is loaded whole to read anything."""
        return self._backend.is_columnar

    @property
    def figure_ids_are_server_ids(self) -> bool:
        """Whether figure ids in this snapshot can be matched against another version's.

        False for video snapshots written as schema v2.0.0, where the ids are positions
        in the table rather than the server's - a comparison that matches on them would
        silently pair unrelated figures. Convert such a snapshot with
        :func:`~supervisely.project.video_project.VideoProject.repack_snapshot` first,
        or match on something else.
        """
        return self._backend.figure_ids_are_server_ids

    def datasets(self) -> List[SnapshotDataset]:
        return self._backend.datasets()

    def iter_items(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        columns: Optional[Sequence[str]] = None,
    ) -> Iterator[List[dict]]:
        """
        Yield the project's items in batches, as dicts keyed by :class:`SnapshotColumn`.

        :param batch_size: Rows per batch. This is what bounds peak memory on a columnar snapshot.
        :param columns: Columns to read. None reads them all; naming them is cheaper.
        """
        return self._backend.iter_items(batch_size, columns)

    def iter_tags(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        columns: Optional[Sequence[str]] = None,
    ) -> Iterator[List[dict]]:
        """
        Yield tag assignments in batches - items, annotation objects and figures alike.

        A tag assignment is an entity, not a property of what it hangs on: it has its own
        id and its own timestamps, so additions, removals and changes are the same set
        operation used for every other entity. ``OWNER_TYPE`` says what it is attached to
        and ``OWNER_ID`` which one.
        """
        return self._backend.iter_tags(batch_size, columns)

    def iter_items_arrow(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        columns: Optional[Sequence[str]] = None,
    ):
        """
        Yield the project's items as ``pyarrow.RecordBatch``, columns named as :class:`SnapshotColumn`.

        The fast path, and the reason the snapshot is Parquet: building a dict per row is
        what costs, not reading the file. On 4.5M figure rows, three columns take 0.13 s
        as Arrow batches against 4.6 s as dicts. Use this for anything that compares,
        hashes or counts whole columns, and the dict interface where per-row Python
        objects are what you actually want.

        Only available on snapshots that store the data as columns - see
        :attr:`is_columnar`. Requesting a column that is derived when rows are built
        (a video figure's class name, for instance) raises rather than lying.

        :param batch_size: Rows per batch.
        :param columns: Columns to read. None reads every column the reader can surface.
        :raises NotImplementedError: the snapshot's items are not stored as columns.
        :raises ValueError: a requested column is derived rather than stored.
        """
        return self._backend.iter_arrow("items", batch_size, columns)

    def iter_figures_arrow(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        columns: Optional[Sequence[str]] = None,
    ):
        """
        Yield the project's figures as ``pyarrow.RecordBatch``. See :func:`iter_items_arrow`.

        For images, ``GEOMETRY`` here is the stored column and is null for alpha masks -
        their payloads live in their own table and are joined only when rows are built.
        """
        return self._backend.iter_arrow("figures", batch_size, columns)

    def iter_figures(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        columns: Optional[Sequence[str]] = None,
        with_geometry: bool = True,
    ) -> Iterator[List[dict]]:
        """
        Yield the project's figures in batches, as dicts keyed by :class:`SnapshotColumn`.

        :param batch_size: Rows per batch.
        :param columns: Columns to read. None reads them all.
        :param with_geometry: Read geometry payloads. Turning it off skips the alpha-mask
            table entirely, which is most of a snapshot's bytes when masks are used.
        """
        return self._backend.iter_figures(batch_size, columns, with_geometry)

    # --------------------------------------------------------------- closing

    def close(self) -> None:
        """Remove whatever this snapshot owns. A cached snapshot's directory is not owned."""
        for directory in self._cleanup_dirs:
            shutil.rmtree(directory, ignore_errors=True)
        self._cleanup_dirs = []

    def __enter__(self) -> "VersionSnapshot":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
