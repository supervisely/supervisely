"""One table for every tag assignment in a snapshot, whatever it hangs on.

The server keeps three tables - ``images_tags``, ``annotation_objects_tags`` and
``figures_tags`` - with near-identical columns, and the API renders all three into the
same shape. So a snapshot needs one table with an owner discriminator, not three.

A tag assignment is an entity in its own right: it has its own id, its own timestamps,
and its own value. Keeping it as JSON text on the row it belongs to - which is what this
format did before - turns a row that can be compared by id into text that has to be
parsed to be compared, and gives the frame bounds no type at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from supervisely.project.versioning.schema_fields import VersionSchemaField

# What the tag hangs on. The server splits these into separate tables; here it is a
# column, dictionary-encoded by Parquet down to almost nothing.
OWNER_ITEM = "item"
OWNER_OBJECT = "object"
OWNER_FIGURE = "figure"


def _as_json(value) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value)


def _as_int(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class TagSchema:
    """The tags table. Same columns for item, object and figure tags."""

    schema_version: str

    def tags_schema(self, pa_module):
        return pa_module.schema(
            [
                # The assignment's own id - what makes a tag addition or removal a set
                # operation rather than a text comparison.
                (VersionSchemaField.TAG_ASSIGNMENT_ID, pa_module.int64()),
                (VersionSchemaField.OWNER_TYPE, pa_module.utf8()),
                (VersionSchemaField.OWNER_ID, pa_module.int64()),
                (VersionSchemaField.SRC_ITEM_ID, pa_module.int64()),
                # The tag meta this is an instance of.
                (VersionSchemaField.TAG_ID, pa_module.int64()),
                (VersionSchemaField.NAME, pa_module.utf8()),
                # JSON-encoded, so a number stays a number and a string stays a string.
                # The server column is varchar and the API types it on the way out, so
                # storing the rendered value verbatim would lose that distinction.
                (VersionSchemaField.VALUE_JSON, pa_module.utf8()),
                # frameRange is two integer columns on the server, and it is two here.
                (VersionSchemaField.FRAME_FROM, pa_module.int32()),
                (VersionSchemaField.FRAME_TO, pa_module.int32()),
                (VersionSchemaField.IS_FINISHED, pa_module.bool_()),
                (VersionSchemaField.NON_FINAL_VALUE, pa_module.bool_()),
                (VersionSchemaField.LABELER_LOGIN, pa_module.utf8()),
                (VersionSchemaField.CUSTOM_DATA, pa_module.utf8()),
                (VersionSchemaField.CREATED_AT, pa_module.utf8()),
                (VersionSchemaField.UPDATED_AT, pa_module.utf8()),
            ]
        )

    def tag_row(
        self,
        tag_json: Dict[str, Any],
        *,
        owner_type: str,
        owner_id: int,
        src_item_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """One row from a tag as the API renders it, for any of the three owners."""
        frame_range = tag_json.get("frameRange") or []
        return {
            VersionSchemaField.TAG_ASSIGNMENT_ID: _as_int(tag_json.get("id")),
            VersionSchemaField.OWNER_TYPE: owner_type,
            VersionSchemaField.OWNER_ID: owner_id,
            # Carried so a comparison can narrow to one item's tags without a join.
            VersionSchemaField.SRC_ITEM_ID: src_item_id if src_item_id is not None else (
                owner_id if owner_type == OWNER_ITEM else None
            ),
            VersionSchemaField.TAG_ID: _as_int(tag_json.get("tagId")),
            VersionSchemaField.NAME: tag_json.get("name"),
            VersionSchemaField.VALUE_JSON: _as_json(tag_json.get("value")),
            VersionSchemaField.FRAME_FROM: _as_int(frame_range[0]) if frame_range else None,
            VersionSchemaField.FRAME_TO: _as_int(frame_range[1]) if len(frame_range) > 1 else None,
            VersionSchemaField.IS_FINISHED: tag_json.get("isFinished"),
            VersionSchemaField.NON_FINAL_VALUE: tag_json.get("nonFinalValue"),
            VersionSchemaField.LABELER_LOGIN: tag_json.get("labelerLogin"),
            VersionSchemaField.CUSTOM_DATA: _as_json(tag_json.get("customData")),
            VersionSchemaField.CREATED_AT: tag_json.get("createdAt"),
            VersionSchemaField.UPDATED_AT: tag_json.get("updatedAt"),
        }


def tag_json_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the tag as the API renders it, for a restore.

    Only the keys that were actually present are emitted: a restore posts this back, and
    a null ``frameRange`` or ``isFinished`` on a tag that never had one is not the same
    as an absent key.
    """
    out: Dict[str, Any] = {"tagId": row.get(VersionSchemaField.TAG_ID)}
    if row.get(VersionSchemaField.TAG_ASSIGNMENT_ID) is not None:
        out["id"] = row[VersionSchemaField.TAG_ASSIGNMENT_ID]
    if row.get(VersionSchemaField.NAME) is not None:
        out["name"] = row[VersionSchemaField.NAME]
    raw_value = row.get(VersionSchemaField.VALUE_JSON)
    out["value"] = json.loads(raw_value) if raw_value is not None else None
    frame_from = row.get(VersionSchemaField.FRAME_FROM)
    if frame_from is not None:
        out["frameRange"] = [frame_from, row.get(VersionSchemaField.FRAME_TO)]
    for column, key in (
        (VersionSchemaField.IS_FINISHED, "isFinished"),
        (VersionSchemaField.NON_FINAL_VALUE, "nonFinalValue"),
        (VersionSchemaField.LABELER_LOGIN, "labelerLogin"),
        (VersionSchemaField.CREATED_AT, "createdAt"),
        (VersionSchemaField.UPDATED_AT, "updatedAt"),
    ):
        if row.get(column) is not None:
            out[key] = row[column]
    custom_data = row.get(VersionSchemaField.CUSTOM_DATA)
    if custom_data:
        out["customData"] = json.loads(custom_data)
    return out


TAGS_TABLE = "tags"

_TAG_SCHEMAS: Dict[str, TagSchema] = {
    "v1.0.0": TagSchema(schema_version="v1.0.0"),
}


def get_tag_schema(schema_version: str = "v1.0.0") -> TagSchema:
    schema = _TAG_SCHEMAS.get(schema_version)
    if schema is None:
        raise RuntimeError(f"Unsupported tag table schema_version: {schema_version!r}")
    return schema
