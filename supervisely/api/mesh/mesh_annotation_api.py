# coding: utf-8
from __future__ import annotations

import json
import uuid
from typing import Callable, Dict, List, Optional, Union

from tqdm import tqdm

from supervisely._utils import batched
from supervisely.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely.api.module_api import ApiField
from supervisely.io.json import load_json_file
from supervisely.mesh_annotation.constants import KEY, MESH_ID
from supervisely.mesh_annotation.mesh_annotation import MeshAnnotation
from supervisely.mesh_annotation.mesh_indices import (
    decode_mesh_indices_in_json,
    encode_mesh_indices_in_json,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.key_id_map import KeyIdMap


class MeshAnnotationAPI(EntityAnnotationAPI):
    """API for downloading and uploading stored mesh annotation JSON."""

    _method_download_bulk = "entities.annotations.bulk.info"
    _method_upload_bulk = "entities.annotations.bulk.add"
    _entity_ids_str = ApiField.ENTITY_IDS

    def download(
        self,
        mesh_id: int,
        project_meta: Optional[ProjectMeta] = None,
        key_id_map: Optional[KeyIdMap] = None,
        decode_mesh_indices: bool = True,
    ) -> Dict:
        """
        Download stored mesh annotation JSON by mesh ID.

        Mesh annotations are expected to be stored as annotation JSON attached to the entity. Mesh
        index arrays are stored in JSON as little-endian uint32 base64 strings and decoded to
        integer lists by default.
        """
        dataset_id = self._get_mesh_dataset_id(mesh_id)
        ann_json = self.download_bulk(
            dataset_id, [mesh_id], decode_mesh_indices=decode_mesh_indices
        )[0]
        self._update_key_id_map(mesh_id, ann_json, key_id_map)
        return ann_json

    def download_bulk(
        self,
        dataset_id: int,
        mesh_ids: List[int],
        decode_mesh_indices: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[Dict]:
        """
        Download stored mesh annotation JSONs by mesh IDs.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param mesh_ids: Mesh entity IDs.
        :type mesh_ids: List[int]
        :param decode_mesh_indices: Decode base64 mesh index fields to integer lists.
        :type decode_mesh_indices: bool
        :param progress_cb: Progress callback.
        :type progress_cb: tqdm or callable, optional
        :returns: Annotation JSONs ordered like ``mesh_ids``.
        :rtype: List[dict]
        """
        annotations = []
        for batch in batched(mesh_ids):
            response = self._api.post(
                self._method_download_bulk,
                {ApiField.DATASET_ID: dataset_id, self._entity_ids_str: batch},
            )
            annotations.extend(
                self._normalize_download_response(response.json(), batch, decode_mesh_indices)
            )
            if progress_cb is not None:
                self._update_progress(progress_cb, len(batch))
        return annotations

    def append(
        self,
        mesh_id: int,
        ann: MeshAnnotation,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> None:
        """
        Store a full mesh annotation JSON for the mesh entity.

        This method intentionally writes the annotation JSON through the entity annotation storage
        endpoint instead of recreating it from generic object, figure, and tag rows.
        """
        if key_id_map is None:
            key_id_map = KeyIdMap()
        key_id_map.add_video(ann.key(), mesh_id)
        self.upload_json(mesh_id, ann.to_json(key_id_map), key_id_map=key_id_map)

    def upload_json(
        self,
        mesh_id: int,
        ann_json: Dict,
        dataset_id: Optional[int] = None,
        key_id_map: Optional[KeyIdMap] = None,
        encode_mesh_indices: bool = True,
    ) -> None:
        """
        Upload stored annotation JSON for a single mesh entity.

        Mesh index arrays under fields like ``indices`` or ``faceIndices`` are encoded to base64
        little-endian uint32 strings before upload.
        """
        self.upload_jsons(
            dataset_id=dataset_id,
            mesh_ids=[mesh_id],
            anns_json=[ann_json],
            key_id_map=key_id_map,
            encode_mesh_indices=encode_mesh_indices,
        )

    def upload_jsons(
        self,
        dataset_id: Optional[int],
        mesh_ids: List[int],
        anns_json: List[Dict],
        key_id_map: Optional[KeyIdMap] = None,
        encode_mesh_indices: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Upload stored annotation JSONs for mesh entities.

        :param dataset_id: Dataset ID. If omitted, it is resolved from the first mesh ID.
        :type dataset_id: int, optional
        :param mesh_ids: Mesh entity IDs.
        :type mesh_ids: List[int]
        :param anns_json: Mesh annotation JSONs.
        :type anns_json: List[dict]
        """
        if len(mesh_ids) != len(anns_json):
            raise ValueError(
                f"mesh_ids and anns_json must have the same length: "
                f"{len(mesh_ids)} != {len(anns_json)}."
            )
        if len(mesh_ids) == 0:
            return

        if dataset_id is None:
            dataset_id = self._get_mesh_dataset_id(mesh_ids[0])

        for batch in batched(list(zip(mesh_ids, anns_json))):
            data = []
            for mesh_id, ann_json in batch:
                prepared_ann = self._prepare_annotation_json(
                    mesh_id, ann_json, key_id_map, encode_mesh_indices
                )
                data.append({ApiField.ENTITY_ID: mesh_id, ApiField.ANNOTATION: prepared_ann})

            self._api.post(
                self._method_upload_bulk,
                {ApiField.DATASET_ID: dataset_id, ApiField.ANNOTATIONS: data},
            )
            if progress_cb is not None:
                self._update_progress(progress_cb, len(batch))

    def upload_path(
        self,
        mesh_id: int,
        ann_path: str,
        dataset_id: Optional[int] = None,
        key_id_map: Optional[KeyIdMap] = None,
        encode_mesh_indices: bool = True,
    ) -> None:
        """Upload a mesh annotation from a local JSON file."""
        self.upload_json(
            mesh_id,
            load_json_file(ann_path),
            dataset_id=dataset_id,
            key_id_map=key_id_map,
            encode_mesh_indices=encode_mesh_indices,
        )

    def upload_paths(
        self,
        dataset_id: Optional[int],
        mesh_ids: List[int],
        ann_paths: List[str],
        key_id_map: Optional[KeyIdMap] = None,
        encode_mesh_indices: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """Upload mesh annotations from local JSON files."""
        if len(mesh_ids) != len(ann_paths):
            raise ValueError(
                f"mesh_ids and ann_paths must have the same length: "
                f"{len(mesh_ids)} != {len(ann_paths)}."
            )
        self.upload_jsons(
            dataset_id=dataset_id,
            mesh_ids=mesh_ids,
            anns_json=[load_json_file(path) for path in ann_paths],
            key_id_map=key_id_map,
            encode_mesh_indices=encode_mesh_indices,
            progress_cb=progress_cb,
        )

    @staticmethod
    def _prepare_annotation_json(
        mesh_id: int,
        ann_json: Dict,
        key_id_map: Optional[KeyIdMap],
        encode_mesh_indices: bool,
    ) -> Dict:
        prepared_ann = dict(ann_json)
        prepared_ann.setdefault(MESH_ID, mesh_id)

        if key_id_map is not None and prepared_ann.get(KEY) is not None:
            try:
                key_id_map.add_video(uuid.UUID(prepared_ann[KEY]), mesh_id)
            except Exception:
                pass

        if encode_mesh_indices:
            prepared_ann = encode_mesh_indices_in_json(prepared_ann)
        return prepared_ann

    def _get_mesh_dataset_id(self, mesh_id: int) -> int:
        mesh_api = getattr(self._api, "mesh", None)
        if mesh_api is not None:
            return mesh_api.get_info_by_id(mesh_id).dataset_id
        info = self._api.post(
            "entities.info",
            {ApiField.ID: mesh_id, ApiField.FIELDS: [ApiField.DATASET_ID]},
        ).json()
        return info[ApiField.DATASET_ID]

    @staticmethod
    def _normalize_download_response(
        response_json,
        mesh_ids: List[int],
        decode_indices: bool,
    ) -> List[Dict]:
        items = MeshAnnotationAPI._extract_download_items(response_json)
        ordered = []
        id_to_ann = {}
        has_entity_ids = False

        for item in items:
            entity_id = MeshAnnotationAPI._extract_entity_id(item)
            ann_json = MeshAnnotationAPI._extract_annotation_json(item)
            if decode_indices:
                ann_json = decode_mesh_indices_in_json(ann_json)
            if entity_id is not None:
                id_to_ann[entity_id] = ann_json
                has_entity_ids = True
            ordered.append(ann_json)

        if has_entity_ids:
            return [id_to_ann.get(mesh_id, {}) for mesh_id in mesh_ids]
        return ordered

    @staticmethod
    def _extract_download_items(response_json) -> List:
        if response_json is None:
            return []
        if isinstance(response_json, list):
            return response_json
        if isinstance(response_json, dict):
            for key in (ApiField.ANNOTATIONS, "entities", "items"):
                if isinstance(response_json.get(key), list):
                    return response_json[key]
            return [response_json]
        return []

    @staticmethod
    def _extract_annotation_json(item) -> Dict:
        ann_json = item
        if isinstance(item, dict) and ApiField.ANNOTATION in item:
            ann_json = item[ApiField.ANNOTATION]
        if isinstance(ann_json, str):
            ann_json = json.loads(ann_json)
        if ann_json is None:
            return {}
        return ann_json

    @staticmethod
    def _extract_entity_id(item) -> Optional[int]:
        if not isinstance(item, dict):
            return None
        for field in (ApiField.ENTITY_ID, MESH_ID):
            value = item.get(field)
            if value is not None:
                return value
        annotation = item.get(ApiField.ANNOTATION)
        if isinstance(annotation, dict):
            return annotation.get(MESH_ID)
        return None

    @staticmethod
    def _update_key_id_map(mesh_id: int, ann_json: Dict, key_id_map: Optional[KeyIdMap]) -> None:
        if key_id_map is None or not isinstance(ann_json, dict) or ann_json.get(KEY) is None:
            return
        try:
            key_id_map.add_video(uuid.UUID(ann_json[KEY]), mesh_id)
        except Exception:
            pass

    @staticmethod
    def _update_progress(progress_cb, value: int) -> None:
        if hasattr(progress_cb, "update") and callable(getattr(progress_cb, "update")):
            progress_cb.update(value)
        else:
            progress_cb(value)
