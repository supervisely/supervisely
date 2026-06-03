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


class MeshAnnotationAPI(EntityAnnotationAPI):
    """API for mesh annotations.

    A mesh annotation is a flat list of labels, each persisted as a single mesh
    object (``api.mesh.object``) referencing its mesh entity directly. The nested
    "figures" entity used by video/pointcloud annotations is not applicable here.
    Object index geometry is stored as a separate blob in geometry storage.

    Like image annotations, server-side IDs are carried inline in the annotation
    JSON (``label["id"]``); there is no ``KeyIdMap``. On upload, objects are
    associated with their labels by order.
    """

    def download(
        self,
        mesh_id: int,
        download_mesh_geometries: bool = True,
    ) -> Dict:
        """
        Download mesh annotation by mesh ID.

        :param mesh_id: Mesh ID in Supervisely.
        :type mesh_id: int
        :param download_mesh_geometries: Download raw mesh index geometry blobs and patch them into
            the annotation JSON when labels reference external geometry storage.
        :type download_mesh_geometries: bool
        :returns: Annotation JSON.
        :rtype: dict
        """
        dataset_id = self._api.mesh.get_info_by_id(mesh_id).dataset_id
        return self.download_bulk(
            dataset_id,
            [mesh_id],
            download_mesh_geometries=download_mesh_geometries,
        )[0]

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
        objects_by_mesh_id = self._api.mesh.object.download(dataset_id, mesh_ids)
        for mesh_id, object_infos in objects_by_mesh_id.items():
            for object_info in object_infos:
                label_json = {
                    KEY: uuid.uuid4().hex,
                    ApiField.ID: object_info.id,
                    LabelJsonFields.OBJ_CLASS_NAME: class_id_to_name.get(object_info.class_id),
                    TAGS: self._convert_tag_rows_to_json(object_info.tags, tag_id_to_name),
                    ApiField.GEOMETRY_TYPE: object_info.geometry_type,
                    ApiField.GEOMETRY: object_info.geometry,
                }
                if object_info.priority is not None:
                    label_json[ApiField.PRIORITY] = object_info.priority
                if object_info.custom_data is not None:
                    label_json[ApiField.CUSTOM_DATA] = object_info.custom_data
                if (
                    download_mesh_geometries
                    and object_info.geometry_type == Mesh.geometry_name()
                    and self._extract_mesh_indices(object_info.geometry) is None
                ):
                    mesh_geometry_refs.append((object_info.id, label_json))
                labels_by_mesh_id.setdefault(mesh_id, []).append(label_json)

        if len(mesh_geometry_refs) != 0:
            object_ids = [object_id for object_id, _ in mesh_geometry_refs]
            indices_batch = self._api.mesh.object.download_indices_batch(object_ids)
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

    def append(self, mesh_id: int, ann: Union[MeshAnnotation, Dict]) -> None:
        """Append a full mesh annotation to the mesh entity."""
        info = self._api.mesh.get_info_by_id(mesh_id)
        ann_obj = self._coerce_annotation(ann, info.project_id)
        self._upload_annotation(mesh_id, info.project_id, ann_obj)

    def upload_json(
        self,
        mesh_id: int,
        ann_json: Dict,
        dataset_id: Optional[int] = None,  # kept for backward compatibility
    ) -> None:
        """Upload one mesh annotation JSON."""
        self.append(mesh_id, ann_json)

    def upload_paths(
        self,
        mesh_ids: List[int],
        ann_paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """Upload mesh annotations from local JSON files."""
        if len(mesh_ids) != len(ann_paths):
            raise ValueError(
                f"mesh_ids and ann_paths must have the same length: "
                f"{len(mesh_ids)} != {len(ann_paths)}."
            )
        for mesh_id, ann_path in zip(mesh_ids, ann_paths):
            self.append(mesh_id, load_json_file(ann_path))
            if progress_cb is not None:
                update_progress(progress_cb, 1)

    def _coerce_annotation(
        self,
        ann: Union[MeshAnnotation, Dict],
        project_id: int,
    ) -> MeshAnnotation:
        if isinstance(ann, MeshAnnotation):
            return ann
        if isinstance(ann, dict):
            prepared_ann = self._prepare_annotation_json(ann)
            project_meta = ProjectMeta.from_json(self._api.project.get_meta(project_id))
            return MeshAnnotation.from_json(prepared_ann, project_meta)
        raise TypeError(f"Unsupported mesh annotation type: {type(ann).__name__}")

    def _upload_annotation(
        self,
        mesh_id: int,
        project_id: int,
        ann: MeshAnnotation,
    ) -> None:
        name_to_class_id = self._api.object_class.get_name_to_id_map(project_id)

        objects_json = []
        mesh_positions = []  # indices into objects_json whose geometry is stored separately
        mesh_indices = []
        for label in ann.labels:
            geometry_json = deepcopy(label.geometry.to_json())
            geometry_json.pop(GEOMETRY_TYPE, None)
            geometry_json.pop(GEOMETRY_SHAPE, None)

            object_json = {
                ApiField.ENTITY_ID: mesh_id,
                ApiField.GEOMETRY_TYPE: label.geometry.geometry_name(),
                TAGS: label.tags.to_json(),
            }
            class_id = name_to_class_id.get(label.obj_class.name)
            if class_id is not None:
                object_json[CLASS_ID] = class_id
            if label.priority is not None:
                object_json[ApiField.PRIORITY] = label.priority
            if label.custom_data:
                object_json[ApiField.CUSTOM_DATA] = label.custom_data

            if isinstance(label.geometry, Mesh):
                # Object index geometry is stored as a separate blob.
                mesh_positions.append(len(objects_json))
                mesh_indices.append(label.geometry.indices)
            else:
                object_json[ApiField.GEOMETRY] = geometry_json

            objects_json.append(object_json)

        # Entity-level (annotation) tags.
        self._api.mesh.tag.append_to_entity(mesh_id, project_id, ann.tags)

        # Create objects; IDs come back ordered like objects_json.
        object_ids = self._api.mesh.object.append_bulk(mesh_id, objects_json)

        # Upload object index geometry blobs to the matching objects (by order).
        if len(mesh_positions) != 0:
            mesh_object_ids = [object_ids[position] for position in mesh_positions]
            self._api.mesh.object.upload_indices_batch(mesh_object_ids, mesh_indices)

    @staticmethod
    def _prepare_annotation_json(ann_json: Dict) -> Dict:
        prepared_ann = dict(ann_json)
        if OBJECTS in prepared_ann or FIGURES in prepared_ann:
            raise RuntimeError(
                "Legacy mesh annotation JSON with 'objects'/'figures' is not supported. "
                "Use the 'labels' mesh annotation schema."
            )
        prepared_ann.setdefault(LABELS, [])
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
