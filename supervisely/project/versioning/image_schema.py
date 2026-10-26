from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from supervisely.project.versioning.schema_fields import VersionSchemaField


def _as_int(value) -> Optional[int]:
    """Numeric API fields are not always numbers.

    ``size`` in particular comes back as a JSON string for some images, and pyarrow
    rejects the whole batch when one row's type does not match the column. A value
    that cannot be read as an integer is stored as null rather than failing the
    snapshot - none of these columns is load-bearing for restore.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_json(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)) and len(value) == 0:
        return None
    return json.dumps(value)


@dataclass(frozen=True)
class ImageSnapshotSchema:
    """PyArrow schemas for image project snapshot tables.

    Four tables. ``datasets`` and ``images`` mirror what the pickle format carried as
    ``dataset_infos`` / ``image_infos``; ``figures`` carries the flattened
    ``Dict[image_id, List[FigureInfo]]``; ``alpha_geometries`` is separate because
    alpha-mask payloads are the heaviest thing in a snapshot and a reader that only
    needs to compare annotations must be able to skip them.

    Tags are not here: a tag assignment has its own id and its own timestamps, so it is
    a row in the shared ``tags`` table rather than text on the row it hangs on.

    Only the fields a restore uses, plus the ones a comparison between two snapshots
    needs, are columns here. Server-derived URLs (``path_original``,
    ``full_storage_url``) and recomputable AI-search state are deliberately absent -
    they are meaningless in a restored project.
    """

    schema_version: str

    def datasets_schema(self, pa_module):
        return pa_module.schema(
            [
                (VersionSchemaField.SRC_DATASET_ID, pa_module.int64()),
                (VersionSchemaField.PARENT_SRC_DATASET_ID, pa_module.int64()),
                (VersionSchemaField.NAME, pa_module.utf8()),
                (VersionSchemaField.FULL_PATH, pa_module.utf8()),
                (VersionSchemaField.DESCRIPTION, pa_module.utf8()),
                (VersionSchemaField.CUSTOM_DATA, pa_module.utf8()),
            ]
        )

    def images_schema(self, pa_module):
        return pa_module.schema(
            [
                (VersionSchemaField.SRC_IMAGE_ID, pa_module.int64()),
                (VersionSchemaField.SRC_DATASET_ID, pa_module.int64()),
                # Overlay labeling interface only. Not restored today - upload_hashes
                # does not take parent ids - but carried so that fixing it later is
                # not a schema change.
                (VersionSchemaField.PARENT_SRC_IMAGE_ID, pa_module.int64()),
                (VersionSchemaField.NAME, pa_module.utf8()),
                (VersionSchemaField.HASH, pa_module.utf8()),
                (VersionSchemaField.LINK, pa_module.utf8()),
                (VersionSchemaField.MIME, pa_module.utf8()),
                (VersionSchemaField.EXT, pa_module.utf8()),
                (VersionSchemaField.SIZE, pa_module.int64()),
                (VersionSchemaField.WIDTH, pa_module.int32()),
                (VersionSchemaField.HEIGHT, pa_module.int32()),
                (VersionSchemaField.CREATED_AT, pa_module.utf8()),
                (VersionSchemaField.UPDATED_AT, pa_module.utf8()),
                (VersionSchemaField.CREATED_BY_ID, pa_module.int64()),
                (VersionSchemaField.DESCRIPTION, pa_module.utf8()),
                (VersionSchemaField.META, pa_module.utf8()),
                # Blob-backed images: the bytes live inside another file at an offset.
                (VersionSchemaField.RELATED_DATA_ID, pa_module.int64()),
                (VersionSchemaField.DOWNLOAD_ID, pa_module.utf8()),
                (VersionSchemaField.OFFSET_START, pa_module.int64()),
                (VersionSchemaField.OFFSET_END, pa_module.int64()),
            ]
        )

    def figures_schema(self, pa_module):
        return pa_module.schema(
            [
                (VersionSchemaField.SRC_FIGURE_ID, pa_module.int64()),
                (VersionSchemaField.SRC_IMAGE_ID, pa_module.int64()),
                (VersionSchemaField.SRC_DATASET_ID, pa_module.int64()),
                # class_id is what restore remaps through the project meta; class_name
                # is what a comparison between two snapshots matches on, because ids
                # are not stable across projects.
                (VersionSchemaField.CLASS_ID, pa_module.int64()),
                (VersionSchemaField.CLASS_NAME, pa_module.utf8()),
                (VersionSchemaField.GEOMETRY_TYPE, pa_module.utf8()),
                # Null for alpha masks - see alpha_geometries_schema.
                (VersionSchemaField.GEOMETRY_JSON, pa_module.utf8()),
                (VersionSchemaField.GEOMETRY_META_JSON, pa_module.utf8()),
                (VersionSchemaField.META, pa_module.utf8()),
                (VersionSchemaField.PRIORITY, pa_module.int32()),
                (VersionSchemaField.CUSTOM_DATA, pa_module.utf8()),
                # The pickle format carried these on every FigureInfo. A comparison
                # between two versions of one project can use them as a cheap first
                # filter - the ids and timestamps are the same server's - so dropping
                # them would have made the new format the poorer of the two.
                (VersionSchemaField.CREATED_AT, pa_module.utf8()),
                (VersionSchemaField.UPDATED_AT, pa_module.utf8()),
            ]
        )

    def alpha_geometries_schema(self, pa_module):
        return pa_module.schema(
            [
                (VersionSchemaField.SRC_FIGURE_ID, pa_module.int64()),
                (VersionSchemaField.GEOMETRY_JSON, pa_module.utf8()),
            ]
        )

    def dataset_row_from_ds_info(self, ds_info, *, full_path: str) -> Dict[str, Any]:
        return {
            VersionSchemaField.SRC_DATASET_ID: ds_info.id,
            VersionSchemaField.PARENT_SRC_DATASET_ID: getattr(ds_info, "parent_id", None),
            VersionSchemaField.NAME: ds_info.name,
            VersionSchemaField.FULL_PATH: full_path,
            VersionSchemaField.DESCRIPTION: getattr(ds_info, "description", None),
            VersionSchemaField.CUSTOM_DATA: _as_json(getattr(ds_info, "custom_data", None)),
        }

    def image_row_from_image_info(self, image_info) -> Dict[str, Any]:
        return {
            VersionSchemaField.SRC_IMAGE_ID: image_info.id,
            VersionSchemaField.SRC_DATASET_ID: image_info.dataset_id,
            VersionSchemaField.PARENT_SRC_IMAGE_ID: _as_int(getattr(image_info, "parent_id", None)),
            VersionSchemaField.NAME: image_info.name,
            VersionSchemaField.HASH: getattr(image_info, "hash", None),
            VersionSchemaField.LINK: getattr(image_info, "link", None),
            VersionSchemaField.MIME: getattr(image_info, "mime", None),
            VersionSchemaField.EXT: getattr(image_info, "ext", None),
            VersionSchemaField.SIZE: _as_int(getattr(image_info, "size", None)),
            VersionSchemaField.WIDTH: _as_int(getattr(image_info, "width", None)),
            VersionSchemaField.HEIGHT: _as_int(getattr(image_info, "height", None)),
            VersionSchemaField.CREATED_AT: getattr(image_info, "created_at", None),
            VersionSchemaField.UPDATED_AT: getattr(image_info, "updated_at", None),
            VersionSchemaField.CREATED_BY_ID: _as_int(getattr(image_info, "created_by", None)),
            VersionSchemaField.DESCRIPTION: getattr(image_info, "description", None),
            VersionSchemaField.META: _as_json(getattr(image_info, "meta", None)),
            VersionSchemaField.RELATED_DATA_ID: _as_int(
                getattr(image_info, "related_data_id", None)
            ),
            VersionSchemaField.DOWNLOAD_ID: getattr(image_info, "download_id", None),
            VersionSchemaField.OFFSET_START: _as_int(getattr(image_info, "offset_start", None)),
            VersionSchemaField.OFFSET_END: _as_int(getattr(image_info, "offset_end", None)),
        }

    def figure_row_from_figure_info(
        self,
        figure_info,
        *,
        src_image_id: int,
        class_name: Optional[str],
        store_geometry: bool,
    ) -> Dict[str, Any]:
        """One figure row.

        ``store_geometry`` is False for alpha masks: their geometry is downloaded
        separately and goes to the alpha_geometries table, keyed by the same figure id.
        """
        return {
            VersionSchemaField.SRC_FIGURE_ID: figure_info.id,
            VersionSchemaField.SRC_IMAGE_ID: src_image_id,
            VersionSchemaField.SRC_DATASET_ID: getattr(figure_info, "dataset_id", None),
            VersionSchemaField.CLASS_ID: _as_int(getattr(figure_info, "class_id", None)),
            VersionSchemaField.CLASS_NAME: class_name,
            VersionSchemaField.GEOMETRY_TYPE: figure_info.geometry_type,
            VersionSchemaField.GEOMETRY_JSON: (
                _as_json(getattr(figure_info, "geometry", None)) if store_geometry else None
            ),
            VersionSchemaField.GEOMETRY_META_JSON: _as_json(
                getattr(figure_info, "geometry_meta", None)
            ),
            VersionSchemaField.META: _as_json(getattr(figure_info, "meta", None)),
            VersionSchemaField.PRIORITY: _as_int(getattr(figure_info, "priority", None)),
            VersionSchemaField.CUSTOM_DATA: _as_json(getattr(figure_info, "custom_data", None)),
            VersionSchemaField.CREATED_AT: getattr(figure_info, "created_at", None),
            VersionSchemaField.UPDATED_AT: getattr(figure_info, "updated_at", None),
        }

    def alpha_geometry_row(self, *, src_figure_id: int, geometry: Optional[dict]):
        return {
            VersionSchemaField.SRC_FIGURE_ID: src_figure_id,
            VersionSchemaField.GEOMETRY_JSON: _as_json(geometry),
        }


# Both entries are the same layout. "v2.1.0" is what new snapshots are written as, so
# that one number means the same thing in every modality: a columnar snapshot whose
# object and figure ids are the server's, and which can therefore be compared with
# another version. "v2.0.0" stays readable - image snapshots written before the
# renumbering carry it, and they are byte-identical in structure.
_IMAGE_SCHEMAS: Dict[str, ImageSnapshotSchema] = {
    "v2.0.0": ImageSnapshotSchema(schema_version="v2.0.0"),
    "v2.1.0": ImageSnapshotSchema(schema_version="v2.1.0"),
}
