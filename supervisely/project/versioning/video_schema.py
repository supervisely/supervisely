from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from supervisely.project.versioning.schema_fields import VersionSchemaField


@dataclass(frozen=True)
class VideoSnapshotSchema:
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

    def videos_schema(self, pa_module):
        return pa_module.schema(
            [
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
                (VersionSchemaField.ANN_JSON, pa_module.utf8()),
            ]
        )

    def objects_schema(self, pa_module):
        return pa_module.schema(
            [
                (VersionSchemaField.SRC_OBJECT_ID, pa_module.int64()),
                (VersionSchemaField.SRC_VIDEO_ID, pa_module.int64()),
                (VersionSchemaField.CLASS_NAME, pa_module.utf8()),
                (VersionSchemaField.KEY, pa_module.utf8()),
                (VersionSchemaField.TAGS_JSON, pa_module.utf8()),
            ]
        )

    def figures_schema(self, pa_module):
        return pa_module.schema(
            [
                (VersionSchemaField.SRC_FIGURE_ID, pa_module.int64()),
                (VersionSchemaField.SRC_OBJECT_ID, pa_module.int64()),
                (VersionSchemaField.SRC_VIDEO_ID, pa_module.int64()),
                (VersionSchemaField.FRAME_INDEX, pa_module.int32()),
                (VersionSchemaField.GEOMETRY_TYPE, pa_module.utf8()),
                (VersionSchemaField.GEOMETRY_JSON, pa_module.utf8()),
            ]
        )

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

_VIDEO_SCHEMAS: Dict[str, VideoSnapshotSchema] = {
    "v2.0.0": VideoSnapshotSchema(schema_version="v2.0.0"),
}

