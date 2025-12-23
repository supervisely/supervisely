from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.geometry.mask_3d import Mask3D
from supervisely.project.data_version import VersionSchemaField
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.versioning.schema_fields import VersionSchemaField
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation


@dataclass(frozen=True)
class VolumeSnapshotSchema:
    schema_version: str

    def datasets_table_schema(self, pa_module):
        return pa_module.schema(
            [
                (VersionSchemaField.SRC_DATASET_ID, pa_module.int64()),
                (VersionSchemaField.JSON, pa_module.large_string()),
            ]
        )

    def volumes_table_schema(self, pa_module):
        return pa_module.schema(
            [
                (VersionSchemaField.SRC_VOLUME_ID, pa_module.int64()),
                (VersionSchemaField.SRC_DATASET_ID, pa_module.int64()),
                (VersionSchemaField.JSON, pa_module.large_string()),
            ]
        )

    def annotations_table_schema(self, pa_module):
        return pa_module.schema(
            [
                (VersionSchemaField.SRC_VOLUME_ID, pa_module.int64()),
                (VersionSchemaField.ANNOTATION, pa_module.large_string()),
            ]
        )

    def dataset_row_from_record(self, dataset_record: dict) -> Dict[str, Any]:
        return {
            VersionSchemaField.SRC_DATASET_ID: dataset_record.get(ApiField.ID),
            VersionSchemaField.JSON: json.dumps(dataset_record, ensure_ascii=False),
        }

    def volume_row_from_record(self, volume_record: dict) -> Dict[str, Any]:
        return {
            VersionSchemaField.SRC_VOLUME_ID: volume_record.get(ApiField.ID),
            VersionSchemaField.SRC_DATASET_ID: volume_record.get(ApiField.DATASET_ID),
            VersionSchemaField.JSON: json.dumps(volume_record, ensure_ascii=False),
        }

    def annotation_row_from_dict(self, *, src_volume_id: int, annotation: dict) -> Dict[str, Any]:
        return {
            VersionSchemaField.SRC_VOLUME_ID: src_volume_id,
            VersionSchemaField.ANNOTATION: json.dumps(annotation, ensure_ascii=False),
        }

    def annotation_dict_from_raw(
        self,
        *,
        api: Api,
        raw_ann_json: dict,
        project_meta_obj: ProjectMeta,
        key_id_map: KeyIdMap,
    ) -> dict:
        ann = VolumeAnnotation.from_json(raw_ann_json, project_meta_obj, key_id_map)
        self._load_mask_geometries(api, ann, key_id_map)
        return ann.to_json()

    @staticmethod
    def _load_mask_geometries(api: Api, ann: VolumeAnnotation, key_id_map: KeyIdMap) -> None:
        for sf in ann.spatial_figures:
            if sf.geometry.name() != Mask3D.name():
                continue
            api.volume.figure.load_sf_geometry(sf, key_id_map)


_VOLUME_SCHEMAS: Dict[str, VolumeSnapshotSchema] = {
    "v2.0.0": VolumeSnapshotSchema(schema_version="v2.0.0"),
}
