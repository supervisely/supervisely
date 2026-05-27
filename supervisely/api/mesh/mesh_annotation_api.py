# coding: utf-8
from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

from tqdm import tqdm

from supervisely.annotation.label import LabelJsonFields
from supervisely.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely.api.module_api import ApiField
from supervisely.geometry.constants import INDICES
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
from supervisely.sly_logger import logger
from supervisely.task.progress import update_progress
from supervisely.video_annotation.key_id_map import KeyIdMap


class MeshAnnotationAPI(EntityAnnotationAPI):
    """API for mesh annotations backed by generic object, figure, and tag rows."""

    def download(
        self,
        mesh_id: int,
        key_id_map: Optional[KeyIdMap] = None,
        download_mesh_geometries: bool = True,
        skip_orphan_objects: bool = False,
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
        :param skip_orphan_objects: If ``True``, skip orphan object rows during download with a warning.
        :type skip_orphan_objects: bool
        :returns: Annotation JSON.
        :rtype: dict
        """
        dataset_id = self._api.mesh.get_info_by_id(mesh_id).dataset_id
        ann_json = self.download_bulk(
            dataset_id,
            [mesh_id],
            download_mesh_geometries=download_mesh_geometries,
            skip_orphan_objects=skip_orphan_objects,
        )[0]
        self._update_key_id_map(mesh_id, ann_json, key_id_map)
        return ann_json

    def download_bulk(
        self,
        dataset_id: int,
        mesh_ids: List[int],
        download_mesh_geometries: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_orphan_objects: bool = False,
    ) -> List[Dict]:
        """
        Download mesh annotation transfer JSONs by mesh IDs.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param mesh_ids: Mesh entity IDs.
        :type mesh_ids: List[int]
        :param download_mesh_geometries: Download raw mesh index geometry blobs and patch them into JSON when annotation labels reference external geometry.
        :type download_mesh_geometries: bool
        :param progress_cb: Progress callback.
        :type progress_cb: tqdm or callable, optional
        :param skip_orphan_objects: If ``True``, skip orphan object rows during download with a warning.
        :type skip_orphan_objects: bool
        :returns: Annotation JSONs ordered like ``mesh_ids``.
        :rtype: List[dict]
        """
        return self._download_bulk_from_entity_rows(
            dataset_id, mesh_ids, download_mesh_geometries, progress_cb, skip_orphan_objects
        )

    def append(
        self,
        mesh_id: int,
        ann: Union[MeshAnnotation, Dict],
        key_id_map: Optional[KeyIdMap] = None,
    ) -> None:
        """
        Append a full mesh annotation to the mesh entity.
        """
        if key_id_map is None:
            key_id_map = KeyIdMap()
        if isinstance(ann, MeshAnnotation):
            key_id_map.add_video(ann.key(), mesh_id)
            ann_json = ann.to_json(key_id_map)
        elif isinstance(ann, dict):
            ann_json = deepcopy(ann)
        else:
            raise TypeError(f"Unsupported mesh annotation type: {type(ann).__name__}")
        dataset_id = self._api.mesh.get_info_by_id(mesh_id).dataset_id
        self._upload_jsons_as_entity_rows(
            dataset_id, [mesh_id], [ann_json], key_id_map=key_id_map
        )

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
        if len(mesh_ids) == 0:
            return

        dataset_id_to_mesh_ids = {}
        project_id_to_mesh_ids = {}
        for mesh_id in mesh_ids:
            mesh_info = self._api.mesh.get_info_by_id(mesh_id)
            dataset_id_to_mesh_ids.setdefault(mesh_info.dataset_id, []).append(mesh_id)
            project_id_to_mesh_ids.setdefault(mesh_info.project_id, []).append(mesh_id)

        if len(dataset_id_to_mesh_ids) != 1 or len(project_id_to_mesh_ids) != 1:
            logger.warning(
                "Can not upload mesh annotations from multiple projects or datasets. "
                f"Project to mesh ids: {project_id_to_mesh_ids}. "
                f"Dataset to mesh ids: {dataset_id_to_mesh_ids}."
            )
            raise RuntimeError("All meshes must belong to the same project and dataset.")

        dataset_id = next(iter(dataset_id_to_mesh_ids))
        self._upload_jsons_as_entity_rows(
            dataset_id,
            mesh_ids,
            [load_json_file(path) for path in ann_paths],
            key_id_map=key_id_map,
            progress_cb=progress_cb,
        )

    def upload_json(
        self,
        mesh_id: int,
        ann_json: Dict,
        dataset_id: Optional[int] = None,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> None:
        """Upload one mesh annotation JSON."""
        if dataset_id is None:
            dataset_id = self._api.mesh.get_info_by_id(mesh_id).dataset_id
        self._upload_jsons_as_entity_rows(
            dataset_id,
            [mesh_id],
            [ann_json],
            key_id_map=key_id_map,
        )

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

    def _upload_jsons_as_entity_rows(
        self,
        dataset_id: int,
        mesh_ids: List[int],
        anns_json: List[Dict],
        key_id_map: Optional[KeyIdMap] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
        project_meta = ProjectMeta.from_json(self._api.project.get_meta(project_id))
        if key_id_map is None:
            key_id_map = KeyIdMap()

        for mesh_id, ann_json in zip(mesh_ids, anns_json):
            prepared_ann = self._prepare_annotation_json(mesh_id, ann_json, key_id_map)
            ann = MeshAnnotation.from_json(prepared_ann, project_meta, key_id_map)

            self._api.mesh.tag.append_to_entity(mesh_id, project_id, ann.tags, key_id_map=key_id_map)
            self._api.mesh.object.append_bulk(mesh_id, ann.labels, key_id_map)
            self._api.mesh.figure.append_bulk(mesh_id, ann.labels, key_id_map)

            if progress_cb is not None:
                update_progress(progress_cb, 1)

    def _download_bulk_from_entity_rows(
        self,
        dataset_id: int,
        mesh_ids: List[int],
        download_mesh_geometries: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_orphan_objects: bool = False,
    ) -> List[Dict]:
        if len(mesh_ids) == 0:
            return []

        project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
        obj_class_id_to_name = {
            obj_class.id: obj_class.name for obj_class in self._api.object_class.get_list(project_id)
        }
        tag_id_to_name = {tag.id: tag.name for tag in self._api.mesh.tag.get_list(project_id)}

        labels_by_mesh_id = {mesh_id: [] for mesh_id in mesh_ids}
        object_ids_by_mesh_id = {mesh_id: set() for mesh_id in mesh_ids}
        used_object_ids_by_mesh_id = {mesh_id: set() for mesh_id in mesh_ids}
        object_id_to_json = {}
        object_infos = self._api.mesh.object.get_list(
            dataset_id,
            filters=[
                {
                    ApiField.FIELD: ApiField.ENTITY_ID,
                    ApiField.OPERATOR: "in",
                    ApiField.VALUE: mesh_ids,
                }
            ],
        )
        for object_info in object_infos:
            object_key = uuid.uuid4()
            object_json = {
                KEY: object_key.hex,
                ApiField.ID: object_info.id,
                LabelJsonFields.OBJ_CLASS_NAME: obj_class_id_to_name.get(object_info.class_id),
                TAGS: self._convert_tag_rows_to_json(object_info.tags, tag_id_to_name),
            }
            if object_info.entity_id is not None:
                object_ids_by_mesh_id.setdefault(object_info.entity_id, set()).add(object_info.id)
            object_id_to_json[object_info.id] = object_json

        mesh_geometry_refs = []
        raw_figures_by_mesh_id = self._api.mesh.figure.download(dataset_id, mesh_ids)
        for mesh_id, figure_infos in raw_figures_by_mesh_id.items():
            for figure_info in figure_infos:
                if figure_info.object_id in used_object_ids_by_mesh_id.setdefault(mesh_id, set()):
                    raise RuntimeError(
                        "Can not download mesh annotation: multiple figure rows reference "
                        "object id={!r} in mesh id={!r}.".format(figure_info.object_id, mesh_id)
                    )
                object_json = object_id_to_json.get(figure_info.object_id)
                if object_json is None:
                    raise RuntimeError(
                        "Can not download mesh annotation: object row with id={!r} "
                        "was not found for figure id={!r}.".format(
                            figure_info.object_id, figure_info.id
                        )
                    )

                figure_json = {
                    KEY: object_json[KEY],
                    ApiField.ID: figure_info.id,
                    LabelJsonFields.OBJ_CLASS_NAME: object_json[LabelJsonFields.OBJ_CLASS_NAME],
                    TAGS: object_json.get(TAGS, []),
                    ApiField.GEOMETRY_TYPE: figure_info.geometry_type,
                    ApiField.GEOMETRY: figure_info.geometry,
                }
                if figure_info.priority is not None:
                    figure_json[ApiField.PRIORITY] = figure_info.priority
                if figure_info.custom_data is not None:
                    figure_json[ApiField.CUSTOM_DATA] = figure_info.custom_data
                if (
                    download_mesh_geometries
                    and figure_json.get(ApiField.GEOMETRY_TYPE) == Mesh.geometry_name()
                    and self._extract_mesh_indices(figure_json.get(ApiField.GEOMETRY)) is None
                ):
                    mesh_geometry_refs.append((figure_info.id, figure_json))
                labels_by_mesh_id.setdefault(mesh_id, []).append(figure_json)
                used_object_ids_by_mesh_id[mesh_id].add(figure_info.object_id)

        if len(mesh_geometry_refs) != 0:
            figure_ids = [figure_id for figure_id, _ in mesh_geometry_refs]
            indices_batch = self._api.mesh.figure.download_indices_batch(figure_ids)
            for (_, figure_json), indices in zip(mesh_geometry_refs, indices_batch):
                figure_json[ApiField.GEOMETRY] = {INDICES: indices}

        annotations = []
        for mesh_id in mesh_ids:
            orphan_object_ids = object_ids_by_mesh_id.get(mesh_id, set()) - used_object_ids_by_mesh_id.get(
                mesh_id, set()
            )
            if len(orphan_object_ids) != 0:
                message = (
                    "Can not download mesh annotation: object rows without matching figures "
                    "found for mesh id={!r}: {!r}.".format(mesh_id, sorted(orphan_object_ids))
                )
                if skip_orphan_objects:
                    logger.warning(message)
                else:
                    raise RuntimeError(message)
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
