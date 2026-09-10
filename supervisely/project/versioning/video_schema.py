from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from supervisely.project.versioning.schema_fields import VersionSchemaField
from supervisely.sly_logger import logger


def _as_str(value) -> Optional[str]:
    """Anything not already a string is stringified; None stays None."""
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _as_int(value) -> Optional[int]:
    """A numeric column that arrives as something else becomes null rather than failing
    the whole snapshot - pyarrow rejects a batch over one bad row."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class VideoSnapshotSchema:
    """PyArrow schemas for the tables of a video project snapshot.

    Two shapes, chosen by ``schema_version``:

    - ``v2.0.0`` keeps each video's whole annotation as an ``ann_json`` string and also
      normalizes it into objects and figures. A restore reads the string; a reader reads
      the tables. The same data twice, and the tables number their rows rather than
      carrying the server's ids.
    - ``v2.1.0`` drops ``ann_json``. The tables carry the server's ids and every field a
      restore needs, and the annotation is rebuilt from them
      (:func:`annotation_json_from_rows`). One representation, no duplication.
    """

    schema_version: str

    @property
    def stores_ann_json(self) -> bool:
        return self.schema_version == "v2.0.0"

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

    def videos_schema(self, pa_module):
        fields = [
            (VersionSchemaField.SRC_VIDEO_ID, pa_module.int64()),
            (VersionSchemaField.SRC_DATASET_ID, pa_module.int64()),
            (VersionSchemaField.NAME, pa_module.utf8()),
            (VersionSchemaField.HASH, pa_module.utf8()),
            (VersionSchemaField.LINK, pa_module.utf8()),
            (VersionSchemaField.FRAMES_COUNT, pa_module.int32()),
            (VersionSchemaField.FRAME_WIDTH, pa_module.int32()),
            (VersionSchemaField.FRAME_HEIGHT, pa_module.int32()),
            (VersionSchemaField.FRAMES_TO_TIMECODES, pa_module.utf8()),
            (VersionSchemaField.META, pa_module.utf8()),
            (VersionSchemaField.CUSTOM_DATA, pa_module.utf8()),
            (VersionSchemaField.CREATED_AT, pa_module.utf8()),
            (VersionSchemaField.UPDATED_AT, pa_module.utf8()),
        ]
        if self.stores_ann_json:
            fields.append((VersionSchemaField.ANN_JSON, pa_module.utf8()))
        else:
            # The annotation's own fields, which only ann_json used to hold: a video's
            # tags (with their frame ranges) and the annotation description.
            fields.append((VersionSchemaField.DESCRIPTION, pa_module.utf8()))
        return pa_module.schema(fields)

    def objects_schema(self, pa_module):
        return pa_module.schema(
            [
                (VersionSchemaField.SRC_OBJECT_ID, pa_module.int64()),
                (VersionSchemaField.SRC_VIDEO_ID, pa_module.int64()),
                (VersionSchemaField.CLASS_NAME, pa_module.utf8()),
                (VersionSchemaField.KEY, pa_module.utf8()),
                (VersionSchemaField.CREATED_AT, pa_module.utf8()),
                (VersionSchemaField.UPDATED_AT, pa_module.utf8()),
            ]
        )

    def figures_schema(self, pa_module):
        fields = [
            (VersionSchemaField.SRC_FIGURE_ID, pa_module.int64()),
            (VersionSchemaField.SRC_OBJECT_ID, pa_module.int64()),
            (VersionSchemaField.SRC_VIDEO_ID, pa_module.int64()),
            (VersionSchemaField.FRAME_INDEX, pa_module.int32()),
            (VersionSchemaField.GEOMETRY_TYPE, pa_module.utf8()),
            (VersionSchemaField.GEOMETRY_JSON, pa_module.utf8()),
        ]
        if not self.stores_ann_json:
            # Everything else a figure carries. Without these the tables are a lossy
            # projection of the annotation and a restore built from them would quietly
            # drop tracking, priority and smart-tool state.
            fields.extend(
                [
                    (VersionSchemaField.META, pa_module.utf8()),
                    # A track id is a uuid string on the wire, not a number.
                    (VersionSchemaField.TRACK_ID, pa_module.utf8()),
                    (VersionSchemaField.PRIORITY, pa_module.int32()),
                    (VersionSchemaField.SMART_TOOL_INPUT, pa_module.utf8()),
                    (VersionSchemaField.NN_CREATED, pa_module.bool_()),
                    (VersionSchemaField.NN_UPDATED, pa_module.bool_()),
                    # v2.0.0 kept these inside ann_json; they are what lets a comparison
                    # skip a figure without hashing its geometry.
                    (VersionSchemaField.CREATED_AT, pa_module.utf8()),
                    (VersionSchemaField.UPDATED_AT, pa_module.utf8()),
                ]
            )
        return pa_module.schema(fields)

    def dataset_row(
        self,
        *,
        src_dataset_id: int,
        parent_src_dataset_id: Optional[int],
        name: str,
        full_path: str,
        description: Optional[str],
        custom_data: Optional[dict],
    ) -> Dict[str, Any]:
        return {
            VersionSchemaField.SRC_DATASET_ID: src_dataset_id,
            VersionSchemaField.PARENT_SRC_DATASET_ID: parent_src_dataset_id,
            VersionSchemaField.NAME: name,
            VersionSchemaField.FULL_PATH: full_path,
            VersionSchemaField.DESCRIPTION: description,
            VersionSchemaField.CUSTOM_DATA: (
                json.dumps(custom_data) if isinstance(custom_data, dict) and len(custom_data) > 0 else None
            ),
        }

    def video_row(
        self,
        *,
        src_video_id: int,
        src_dataset_id: int,
        name: str,
        hash: Optional[str],
        link: Optional[str],
        frames_count: Optional[int],
        frame_width: Optional[int],
        frame_height: Optional[int],
        frames_to_timecodes: Optional[list],
        meta: Optional[dict],
        custom_data: Optional[dict],
        created_at: Optional[str],
        updated_at: Optional[str],
        ann_json: dict,
    ) -> Dict[str, Any]:
        return {
            VersionSchemaField.SRC_VIDEO_ID: src_video_id,
            VersionSchemaField.SRC_DATASET_ID: src_dataset_id,
            VersionSchemaField.NAME: name,
            VersionSchemaField.HASH: hash,
            VersionSchemaField.LINK: link,
            VersionSchemaField.FRAMES_COUNT: frames_count,
            VersionSchemaField.FRAME_WIDTH: frame_width,
            VersionSchemaField.FRAME_HEIGHT: frame_height,
            VersionSchemaField.FRAMES_TO_TIMECODES: (
                json.dumps(frames_to_timecodes) if frames_to_timecodes else None
            ),
            VersionSchemaField.META: json.dumps(meta) if meta else None,
            VersionSchemaField.CUSTOM_DATA: json.dumps(custom_data) if custom_data else None,
            VersionSchemaField.CREATED_AT: created_at,
            VersionSchemaField.UPDATED_AT: updated_at,
            VersionSchemaField.ANN_JSON: json.dumps(ann_json),
        }

    def object_row(
        self,
        *,
        src_object_id: int,
        src_video_id: int,
        class_name: str,
        key_hex: str,
        tags_json: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        return {
            VersionSchemaField.SRC_OBJECT_ID: src_object_id,
            VersionSchemaField.SRC_VIDEO_ID: src_video_id,
            VersionSchemaField.CLASS_NAME: class_name,
            VersionSchemaField.KEY: key_hex,
            VersionSchemaField.TAGS_JSON: json.dumps(tags_json) if tags_json is not None else None,
        }

    def figure_row(
        self,
        *,
        src_figure_id: int,
        src_object_id: int,
        src_video_id: int,
        frame_index: int,
        geometry_type: str,
        geometry_json: dict,
    ) -> Dict[str, Any]:
        return {
            VersionSchemaField.SRC_FIGURE_ID: src_figure_id,
            VersionSchemaField.SRC_OBJECT_ID: src_object_id,
            VersionSchemaField.SRC_VIDEO_ID: src_video_id,
            VersionSchemaField.FRAME_INDEX: frame_index,
            VersionSchemaField.GEOMETRY_TYPE: geometry_type,
            VersionSchemaField.GEOMETRY_JSON: json.dumps(geometry_json),
        }

    def video_row_from_json(
        self, video_info, *, src_dataset_id: int, ann_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """One video row for v2.1.0: the video's fields plus the annotation's own two."""
        from supervisely.video_annotation import constants as video_constants

        row = self.video_row_from_video_info(
            video_info, src_dataset_id=src_dataset_id, ann_json={}
        )
        row.pop(VersionSchemaField.ANN_JSON, None)
        row[VersionSchemaField.DESCRIPTION] = ann_json.get(video_constants.DESCRIPTION)
        return row

    def object_row_from_json(self, obj_json: Dict[str, Any], *, src_video_id: int) -> Dict[str, Any]:
        """One object row, straight off the server's annotation.

        Schema v2.1.0 onwards. The server identifies an annotation object by its numeric
        id and a figure points at its parent by that same id, so the annotation can be
        read as it arrived: no uuid keys, no KeyIdMap, and ``src_object_id`` is the real
        object id rather than a row counter that shifts whenever anything is inserted.
        """
        from supervisely.annotation.label import LabelJsonFields
        from supervisely.api.module_api import ApiField
        from supervisely.video_annotation import constants as video_constants

        return {
            VersionSchemaField.SRC_OBJECT_ID: obj_json.get(video_constants.ID),
            VersionSchemaField.SRC_VIDEO_ID: src_video_id,
            VersionSchemaField.CLASS_NAME: obj_json.get(LabelJsonFields.OBJ_CLASS_NAME),
            # The server does not store uuid keys; this is null unless one was sent.
            VersionSchemaField.KEY: obj_json.get(video_constants.KEY),
            VersionSchemaField.CREATED_AT: obj_json.get(ApiField.CREATED_AT),
            VersionSchemaField.UPDATED_AT: obj_json.get(ApiField.UPDATED_AT),
        }

    def figure_row_from_json(
        self, figure_json: Dict[str, Any], *, src_video_id: int, frame_index: int
    ) -> Dict[str, Any]:
        """One figure row, straight off the server's annotation. Schema v2.1.0 onwards."""
        from supervisely.api.module_api import ApiField
        from supervisely.video_annotation import constants as video_constants

        row = {
            VersionSchemaField.SRC_FIGURE_ID: figure_json.get(video_constants.ID),
            VersionSchemaField.SRC_OBJECT_ID: figure_json.get(video_constants.OBJECT_ID),
            VersionSchemaField.SRC_VIDEO_ID: src_video_id,
            VersionSchemaField.FRAME_INDEX: frame_index,
            VersionSchemaField.GEOMETRY_TYPE: figure_json.get(ApiField.GEOMETRY_TYPE),
            VersionSchemaField.GEOMETRY_JSON: json.dumps(figure_json.get(ApiField.GEOMETRY)),
        }
        if not self.stores_ann_json:
            meta = figure_json.get(ApiField.META)
            smart_tool = figure_json.get(ApiField.SMART_TOOL_INPUT)
            row.update(
                {
                    VersionSchemaField.META: json.dumps(meta) if meta else None,
                    VersionSchemaField.TRACK_ID: _as_str(figure_json.get(ApiField.TRACK_ID)),
                    VersionSchemaField.PRIORITY: _as_int(figure_json.get(ApiField.PRIORITY)),
                    VersionSchemaField.SMART_TOOL_INPUT: (
                        json.dumps(smart_tool) if smart_tool else None
                    ),
                    VersionSchemaField.NN_CREATED: bool(figure_json.get(ApiField.NN_CREATED)),
                    VersionSchemaField.NN_UPDATED: bool(figure_json.get(ApiField.NN_UPDATED)),
                    VersionSchemaField.CREATED_AT: figure_json.get(ApiField.CREATED_AT),
                    VersionSchemaField.UPDATED_AT: figure_json.get(ApiField.UPDATED_AT),
                }
            )
        return row

    def dataset_row_from_ds_info(self, ds_info, *, full_path: str, custom_data: Optional[dict]):
        return self.dataset_row(
            src_dataset_id=ds_info.id,
            parent_src_dataset_id=ds_info.parent_id,
            name=ds_info.name,
            full_path=full_path,
            description=getattr(ds_info, "description", None),
            custom_data=custom_data,
        )

    def video_row_from_video_info(self, video_info, *, src_dataset_id: int, ann_json: dict):
        return self.video_row(
            src_video_id=video_info.id,
            src_dataset_id=src_dataset_id,
            name=video_info.name,
            hash=getattr(video_info, "hash", None),
            link=getattr(video_info, "link", None),
            frames_count=getattr(video_info, "frames_count", None),
            frame_width=getattr(video_info, "frame_width", None),
            frame_height=getattr(video_info, "frame_height", None),
            frames_to_timecodes=getattr(video_info, "frames_to_timecodes", None),
            meta=getattr(video_info, "meta", None),
            custom_data=getattr(video_info, "custom_data", None),
            created_at=getattr(video_info, "created_at", None),
            updated_at=getattr(video_info, "updated_at", None),
            ann_json=ann_json,
        )

    def object_row_from_object(self, obj, *, src_object_id: int, src_video_id: int) -> Dict[str, Any]:
        return self.object_row(
            src_object_id=src_object_id,
            src_video_id=src_video_id,
            class_name=obj.obj_class.name,
            key_hex=obj.key().hex,
            tags_json=obj.tags.to_json() if getattr(obj, "tags", None) is not None else None,
        )

    def figure_row_from_figure(
        self,
        fig,
        *,
        figure_row_idx: int,
        src_object_id: int,
        src_video_id: int,
        frame_index: int,
    ) -> Dict[str, Any]:
        return self.figure_row(
            src_figure_id=figure_row_idx + 1,
            src_object_id=src_object_id,
            src_video_id=src_video_id,
            frame_index=frame_index,
            geometry_type=fig.geometry.geometry_name(),
            geometry_json=fig.geometry.to_json(),
        )

# v2.0.0 numbered objects and figures by their position in the table and recorded the
# real ids only in key_id_map.json, keyed by a uuid the SDK invented at snapshot time.
# v2.1.0 stores the server's own ids in those columns, which is what lets two versions
# be compared, and drops key_id_map.json - nothing read it.
_VIDEO_SCHEMAS: Dict[str, VideoSnapshotSchema] = {
    "v2.0.0": VideoSnapshotSchema(schema_version="v2.0.0"),
    "v2.1.0": VideoSnapshotSchema(schema_version="v2.1.0"),
}



def annotation_json_from_rows(
    video_row: Dict[str, Any],
    object_rows: List[Dict[str, Any]],
    figure_rows: List[Dict[str, Any]],
    item_tags: Optional[List[Dict[str, Any]]] = None,
    object_tags: Optional[Dict[Any, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Rebuild one video's annotation from the snapshot's tables (schema v2.1.0+).

    This is what replaces the ``ann_json`` column: the tables hold the same information,
    so a restore can put the annotation back together instead of the snapshot carrying it
    twice.

    Figures are linked to their objects by **key**, not by id. The ids in the snapshot
    belong to the project the version was taken from; a restore creates a new project, so
    fresh uuids are minted here and both sides of the link refer to those.
    """
    import uuid as uuid_module

    from supervisely.annotation.label import LabelJsonFields
    from supervisely.api.module_api import ApiField
    from supervisely.video_annotation import constants as video_constants

    def loads(value, default):
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return default
        return default if parsed is None else parsed

    key_by_object_id: Dict[Any, str] = {}
    objects = []
    for row in object_rows:
        key = row.get(VersionSchemaField.KEY) or uuid_module.uuid4().hex
        key_by_object_id[row.get(VersionSchemaField.SRC_OBJECT_ID)] = key
        objects.append(
            {
                video_constants.KEY: key,
                LabelJsonFields.OBJ_CLASS_NAME: row.get(VersionSchemaField.CLASS_NAME),
                LabelJsonFields.TAGS: (object_tags or {}).get(
                    row.get(VersionSchemaField.SRC_OBJECT_ID), []
                ),
            }
        )

    figures_by_frame: Dict[Any, List[dict]] = {}
    for row in figure_rows:
        object_key = key_by_object_id.get(row.get(VersionSchemaField.SRC_OBJECT_ID))
        if object_key is None:
            # A figure whose object is missing cannot be uploaded; dropping it silently
            # would make the restore look complete when it is not.
            logger.warning(
                f"Figure {row.get(VersionSchemaField.SRC_FIGURE_ID)} references object "
                f"{row.get(VersionSchemaField.SRC_OBJECT_ID)}, which is not in the snapshot"
            )
            continue
        figure = {
            video_constants.KEY: uuid_module.uuid4().hex,
            video_constants.OBJECT_KEY: object_key,
            ApiField.GEOMETRY_TYPE: row.get(VersionSchemaField.GEOMETRY_TYPE),
            ApiField.GEOMETRY: loads(row.get(VersionSchemaField.GEOMETRY_JSON), {}),
        }
        meta = loads(row.get(VersionSchemaField.META), None)
        if meta:
            figure[ApiField.META] = meta
        smart_tool = loads(row.get(VersionSchemaField.SMART_TOOL_INPUT), None)
        if smart_tool:
            figure[ApiField.SMART_TOOL_INPUT] = smart_tool
        for column, field in (
            (VersionSchemaField.TRACK_ID, ApiField.TRACK_ID),
            (VersionSchemaField.PRIORITY, ApiField.PRIORITY),
        ):
            if row.get(column) is not None:
                figure[field] = row[column]
        for column, field in (
            (VersionSchemaField.NN_CREATED, ApiField.NN_CREATED),
            (VersionSchemaField.NN_UPDATED, ApiField.NN_UPDATED),
        ):
            if row.get(column):
                figure[field] = True
        figures_by_frame.setdefault(row.get(VersionSchemaField.FRAME_INDEX), []).append(figure)

    return {
        video_constants.IMG_SIZE: {
            "height": video_row.get(VersionSchemaField.FRAME_HEIGHT),
            "width": video_row.get(VersionSchemaField.FRAME_WIDTH),
        },
        video_constants.DESCRIPTION: video_row.get(VersionSchemaField.DESCRIPTION) or "",
        video_constants.KEY: uuid_module.uuid4().hex,
        video_constants.TAGS: item_tags or [],
        video_constants.OBJECTS: objects,
        video_constants.FRAMES: [
            {video_constants.INDEX: index, video_constants.FIGURES: figures}
            for index, figures in sorted(figures_by_frame.items())
        ],
        video_constants.FRAMES_COUNT: video_row.get(VersionSchemaField.FRAMES_COUNT),
    }
