# coding: utf-8
from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

from tqdm import tqdm

from supervisely.annotation.label import LabelJsonFields
from supervisely.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely.api.module_api import ApiField
from supervisely.geometry.constants import (
    CLASS_ID,
    GEOMETRY_SHAPE,
    GEOMETRY_TYPE,
    INDICES,
)
from supervisely.geometry.mesh import Mesh
from supervisely.io.json import load_json_file
from supervisely.mesh_annotation.constants import (
    FIGURES,
    KEY,
    LABELS,
    MESH_ID,
    OBJECTS,
    TAGS,
)
from supervisely.mesh_annotation.mesh_annotation import MeshAnnotation
from supervisely.mesh_annotation.mesh_indices import MESH_INDEX_FIELDS
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import update_progress
from supervisely.video_annotation.key_id_map import KeyIdMap


class MeshAnnotationAPI(EntityAnnotationAPI):
    """API for mesh annotations.

    Follows the **image** annotation model rather than the video/pointcloud one:
    a mesh annotation is a flat list of labels, each persisted as a single mesh
    object (``api.mesh.object`` — an image-style figure referencing ``entityId`` +
    ``classId``, no separate annotation-object rows). Mesh index geometry is stored
    as a separate blob in geometry storage, analogous to alpha-mask geometry.
    """

    def download(
        self,
        mesh_id: int,
        key_id_map: Optional[KeyIdMap] = None,
        download_mesh_geometries: bool = True,
    ) -> Dict:
        """
        Download mesh annotation by mesh ID.

        :param mesh_id: Mesh ID in Supervisely.
        :type mesh_id: int
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`~supervisely.video_annotation.key_id_map.KeyIdMap`, optional
        :param download_mesh_geometries: Download raw mesh index geometry blobs and patch them into
            the annotation JSON when labels reference external geometry storage.
        :type download_mesh_geometries: bool
        :returns: Annotation JSON.
        :rtype: dict
        """
        dataset_id = self._api.mesh.get_info_by_id(mesh_id).dataset_id
        ann_json = self.download_bulk(
            dataset_id,
            [mesh_id],
            download_mesh_geometries=download_mesh_geometries,
        )[0]
        self._update_key_id_map(mesh_id, ann_json, key_id_map)
        return ann_json

    def download_bulk(
        self,
        dataset_id: int,
        mesh_ids: List[int],
        download_mesh_geometries: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[Dict]:
        """
        Download mesh annotation transfer JSONs by mesh IDs.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param mesh_ids: Mesh entity IDs.
        :type mesh_ids: List[int]
        :param download_mesh_geometries: Download raw mesh index geometry blobs and patch them into
            JSON when annotation labels reference external geometry.
        :type download_mesh_geometries: bool
        :param progress_cb: Progress callback.
        :type progress_cb: tqdm or callable, optional
        :returns: Annotation JSONs ordered like ``mesh_ids``.
        :rtype: List[dict]
        """
        if len(mesh_ids) == 0:
            return []

        project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
        class_id_to_name = {
            obj_class.id: obj_class.name for obj_class in self._api.object_class.get_list(project_id)
        }
        tag_id_to_name = {tag.id: tag.name for tag in self._api.mesh.tag.get_list(project_id)}

        labels_by_mesh_id = {mesh_id: [] for mesh_id in mesh_ids}
        mesh_geometry_refs = []
        figures_by_mesh_id = self._api.mesh.object.download(dataset_id, mesh_ids)
        for mesh_id, figure_infos in figures_by_mesh_id.items():
            for figure_info in figure_infos:
                label_json = {
                    KEY: uuid.uuid4().hex,
                    ApiField.ID: figure_info.id,
                    LabelJsonFields.OBJ_CLASS_NAME: class_id_to_name.get(figure_info.class_id),
                    TAGS: self._convert_tag_rows_to_json(figure_info.tags, tag_id_to_name),
                    ApiField.GEOMETRY_TYPE: figure_info.geometry_type,
                    ApiField.GEOMETRY: figure_info.geometry,
                }
                if figure_info.priority is not None:
                    label_json[ApiField.PRIORITY] = figure_info.priority
                if figure_info.custom_data is not None:
                    label_json[ApiField.CUSTOM_DATA] = figure_info.custom_data
                if (
                    download_mesh_geometries
                    and figure_info.geometry_type == Mesh.geometry_name()
                    and self._extract_mesh_indices(figure_info.geometry) is None
                ):
                    mesh_geometry_refs.append((figure_info.id, label_json))
                labels_by_mesh_id.setdefault(mesh_id, []).append(label_json)

        if len(mesh_geometry_refs) != 0:
            figure_ids = [figure_id for figure_id, _ in mesh_geometry_refs]
            indices_batch = self._api.mesh.object.download_indices_batch(figure_ids)
            for (_, label_json), indices in zip(mesh_geometry_refs, indices_batch):
                label_json[ApiField.GEOMETRY] = {INDICES: indices}

        annotations = []
        for mesh_id in mesh_ids:
            annotations.append(
                {
                    KEY: uuid.uuid4().hex,
                    MESH_ID: mesh_id,
                    TAGS: [],
                    LABELS: labels_by_mesh_id.get(mesh_id, []),
                }
            )
            if progress_cb is not None:
                update_progress(progress_cb, 1)
        return annotations

    def append(
        self,
        mesh_id: int,
        ann: Union[MeshAnnotation, Dict],
        key_id_map: Optional[KeyIdMap] = None,
    ) -> None:
        """Append a full mesh annotation to the mesh entity."""
        if key_id_map is None:
            key_id_map = KeyIdMap()
        info = self._api.mesh.get_info_by_id(mesh_id)
        ann_obj = self._coerce_annotation(ann, info.project_id, mesh_id, key_id_map)
        self._upload_annotation(mesh_id, info.project_id, ann_obj, key_id_map)

    def upload_json(
        self,
        mesh_id: int,
        ann_json: Dict,
        dataset_id: Optional[int] = None,  # kept for backward compatibility
        key_id_map: Optional[KeyIdMap] = None,
    ) -> None:
        """Upload one mesh annotation JSON."""
        self.append(mesh_id, ann_json, key_id_map=key_id_map)

    def upload_paths(
        self,
        mesh_ids: List[int],
        ann_paths: List[str],
        key_id_map: Optional[KeyIdMap] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """Upload mesh annotations from local JSON files."""
        if len(mesh_ids) != len(ann_paths):
            raise ValueError(
                f"mesh_ids and ann_paths must have the same length: "
                f"{len(mesh_ids)} != {len(ann_paths)}."
            )
        if key_id_map is None:
            key_id_map = KeyIdMap()
        for mesh_id, ann_path in zip(mesh_ids, ann_paths):
            self.append(mesh_id, load_json_file(ann_path), key_id_map=key_id_map)
            if progress_cb is not None:
                update_progress(progress_cb, 1)

    def _coerce_annotation(
        self,
        ann: Union[MeshAnnotation, Dict],
        project_id: int,
        mesh_id: int,
        key_id_map: KeyIdMap,
    ) -> MeshAnnotation:
        if isinstance(ann, MeshAnnotation):
            key_id_map.add_video(ann.key(), mesh_id)
            return ann
        if isinstance(ann, dict):
            prepared_ann = self._prepare_annotation_json(mesh_id, ann, key_id_map)
            project_meta = ProjectMeta.from_json(self._api.project.get_meta(project_id))
            return MeshAnnotation.from_json(prepared_ann, project_meta, key_id_map)
        raise TypeError(f"Unsupported mesh annotation type: {type(ann).__name__}")

    def _upload_annotation(
        self,
        mesh_id: int,
        project_id: int,
        ann: MeshAnnotation,
        key_id_map: KeyIdMap,
    ) -> None:
        name_to_class_id = self._api.object_class.get_name_to_id_map(project_id)

        figures_json = []
        figures_keys = []
        mesh_keys = []
        mesh_indices = []
        for label in ann.labels:
            geometry_json = deepcopy(label.geometry.to_json())
            geometry_json.pop(GEOMETRY_TYPE, None)
            geometry_json.pop(GEOMETRY_SHAPE, None)

            figure_json = {
                ApiField.ENTITY_ID: mesh_id,
                ApiField.GEOMETRY_TYPE: label.geometry.geometry_name(),
                TAGS: label.tags.to_json(key_id_map),
            }
            class_id = name_to_class_id.get(label.obj_class.name)
            if class_id is not None:
                figure_json[CLASS_ID] = class_id
            if label.priority is not None:
                figure_json[ApiField.PRIORITY] = label.priority
            if label.custom_data:
                figure_json[ApiField.CUSTOM_DATA] = label.custom_data

            if isinstance(label.geometry, Mesh):
                # Mesh index geometry is stored as a separate blob (like alpha mask).
                mesh_keys.append(label.key())
                mesh_indices.append(label.geometry.indices)
            else:
                figure_json[ApiField.GEOMETRY] = geometry_json

            figures_json.append(figure_json)
            figures_keys.append(label.key())

        # Entity-level (annotation) tags.
        self._api.mesh.tag.append_to_entity(mesh_id, project_id, ann.tags, key_id_map=key_id_map)

        # Create objects (image-style figures) and map label keys -> figure IDs.
        self._api.mesh.object.append_bulk(mesh_id, figures_json, figures_keys, key_id_map)

        # Upload mesh index geometry blobs to the created objects.
        mesh_figure_ids = [key_id_map.get_figure_id(key) for key in mesh_keys]
        if len(mesh_figure_ids) != 0:
            self._api.mesh.object.upload_indices_batch(mesh_figure_ids, mesh_indices)

    @staticmethod
    def _prepare_annotation_json(
        mesh_id: int,
        ann_json: Dict,
        key_id_map: Optional[KeyIdMap],
    ) -> Dict:
        prepared_ann = dict(ann_json)
        if OBJECTS in prepared_ann or FIGURES in prepared_ann:
            raise RuntimeError(
                "Legacy mesh annotation JSON with 'objects'/'figures' is not supported. "
                "Use the 'labels' mesh annotation schema."
            )
        prepared_ann.setdefault(MESH_ID, mesh_id)
        prepared_ann.setdefault(LABELS, [])
        if key_id_map is not None and prepared_ann.get(KEY) is not None:
            try:
                key_id_map.add_video(uuid.UUID(prepared_ann[KEY]), mesh_id)
            except Exception:
                pass
        return prepared_ann

    @staticmethod
    def _extract_mesh_indices(geometry) -> Optional[List[int]]:
        if not isinstance(geometry, dict):
            return None
        for field_name in MESH_INDEX_FIELDS:
            indices = geometry.get(field_name)
            if isinstance(indices, list):
                return indices
        return None

    @staticmethod
    def _convert_tag_rows_to_json(tag_rows: Optional[List[Dict]], tag_id_to_name: Dict[int, str]):
        result = []
        for tag_row in tag_rows or []:
            if not isinstance(tag_row, dict):
                continue
            tag_name = tag_id_to_name.get(tag_row.get(ApiField.TAG_ID))
            if tag_name is None:
                continue
            tag_json = {ApiField.NAME: tag_name}
            if ApiField.VALUE in tag_row:
                tag_json[ApiField.VALUE] = tag_row[ApiField.VALUE]
            if ApiField.ID in tag_row:
                tag_json[ApiField.ID] = tag_row[ApiField.ID]
            result.append(tag_json)
        return result

    @staticmethod
    def _update_key_id_map(mesh_id: int, ann_json: Dict, key_id_map: Optional[KeyIdMap]) -> None:
        if key_id_map is None or not isinstance(ann_json, dict) or ann_json.get(KEY) is None:
            return
        try:
            key_id_map.add_video(uuid.UUID(ann_json[KEY]), mesh_id)
        except Exception:
            pass
        for label_json in ann_json.get(LABELS, []):
            if not isinstance(label_json, dict) or label_json.get(KEY) is None:
                continue
            try:
                label_key = uuid.UUID(label_json[KEY])
                label_id = label_json.get(ApiField.ID)
                if label_id is not None:
                    key_id_map.add_figure(label_key, label_id)
            except Exception:
                pass
