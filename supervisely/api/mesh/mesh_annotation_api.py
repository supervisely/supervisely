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
    MESH_ID,
    OBJECT_KEY,
    OBJECTS,
    TAGS,
)
from supervisely.mesh_annotation.mesh_annotation import MeshAnnotation
from supervisely.mesh_annotation.mesh_indices import MESH_INDEX_FIELDS
from supervisely.mesh_annotation.mesh_object_collection import MeshObjectCollection
from supervisely.mesh_annotation.mesh_tag_collection import MeshTagCollection
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.key_id_map import KeyIdMap


class MeshAnnotationAPI(EntityAnnotationAPI):
    """API for mesh annotations backed by generic object, figure, and tag rows."""

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
            the annotation JSON when figures reference external geometry storage.
        :type download_mesh_geometries: bool
        :returns: Annotation JSON.
        :rtype: dict
        """
        dataset_id = self._api.mesh.get_info_by_id(mesh_id).dataset_id
        ann_json = self.download_bulk(
            dataset_id, [mesh_id], download_mesh_geometries=download_mesh_geometries
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
        :param download_mesh_geometries: Download raw mesh index geometry blobs and patch them into JSON when annotation figures reference external geometry.
        :type download_mesh_geometries: bool
        :param progress_cb: Progress callback.
        :type progress_cb: tqdm or callable, optional
        :returns: Annotation JSONs ordered like ``mesh_ids``.
        :rtype: List[dict]
        """
        return self._download_bulk_from_entity_rows(
            dataset_id, mesh_ids, download_mesh_geometries, progress_cb
        )

    def append(
        self,
        mesh_id: int,
        ann: MeshAnnotation,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> None:
        """
        Append a full mesh annotation to the mesh entity.
        """
        if key_id_map is None:
            key_id_map = KeyIdMap()
        key_id_map.add_video(ann.key(), mesh_id)
        dataset_id = self._api.mesh.get_info_by_id(mesh_id).dataset_id
        self._upload_jsons_as_entity_rows(
            dataset_id, [mesh_id], [ann.to_json(key_id_map)], key_id_map=key_id_map
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
        dataset_id = self._api.mesh.get_info_by_id(mesh_ids[0]).dataset_id
        self._upload_jsons_as_entity_rows(
            dataset_id,
            mesh_ids,
            [load_json_file(path) for path in ann_paths],
            key_id_map=key_id_map,
            progress_cb=progress_cb,
        )

    @staticmethod
    def _prepare_annotation_json(
        mesh_id: int,
        ann_json: Dict,
        key_id_map: Optional[KeyIdMap],
    ) -> Dict:
        prepared_ann = dict(ann_json)
        prepared_ann.setdefault(MESH_ID, mesh_id)
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

            tags = MeshTagCollection.from_json(prepared_ann.get(TAGS, []), project_meta.tag_metas)
            self._api.mesh.tag.append_to_entity(mesh_id, project_id, tags, key_id_map=key_id_map)

            objects = MeshObjectCollection.from_json(prepared_ann.get(OBJECTS, []), project_meta)
            self._api.mesh.object.append_bulk(mesh_id, objects, key_id_map)

            figures_json, figure_keys, indices_by_key = self._prepare_figures_for_entity_rows(
                prepared_ann.get(FIGURES, []), key_id_map
            )
            self._api.mesh.figure._append_bulk(mesh_id, figures_json, figure_keys, key_id_map)
            upload_figure_ids = []
            upload_indices = []
            for figure_key in figure_keys:
                indices = indices_by_key.get(figure_key)
                if indices is None:
                    continue
                upload_figure_ids.append(key_id_map.get_figure_id(figure_key))
                upload_indices.append(indices)
            if len(upload_figure_ids) != 0:
                self._api.mesh.figure.upload_indices_batch(upload_figure_ids, upload_indices)

            if progress_cb is not None:
                self._update_progress(progress_cb, 1)

    def _prepare_figures_for_entity_rows(
        self, figures_json: List[Dict], key_id_map: KeyIdMap
    ) -> tuple:
        prepared_figures = []
        figure_keys = []
        indices_by_key = {}

        for figure_json in figures_json:
            if not isinstance(figure_json, dict):
                continue

            prepared = deepcopy(figure_json)
            figure_key = self._uuid_from_value(prepared.get(KEY)) or uuid.uuid4()
            prepared[KEY] = figure_key.hex
            figure_keys.append(figure_key)

            object_id = prepared.get(ApiField.OBJECT_ID)
            object_key = self._uuid_from_value(prepared.get(OBJECT_KEY))
            if object_key is not None:
                resolved = key_id_map.get_object_id(object_key)
                if resolved is not None:
                    object_id = resolved
            if object_id is None:
                raise RuntimeError(
                    "Can not upload mesh figure: objectId or resolvable objectKey is required."
                )
            prepared[ApiField.OBJECT_ID] = object_id
            prepared.pop(OBJECT_KEY, None)

            geometry_type = prepared.get(ApiField.GEOMETRY_TYPE)
            geometry = prepared.get(ApiField.GEOMETRY) or {}
            if isinstance(geometry, dict):
                geometry = dict(geometry)
                for field_name in MESH_INDEX_FIELDS:
                    geometry.pop(f"{field_name}Path", None)
            prepared[ApiField.GEOMETRY] = geometry

            if geometry_type in (Mesh.geometry_name(), "mesh_indices"):
                prepared[ApiField.GEOMETRY_TYPE] = Mesh.geometry_name()
                indices = self._extract_mesh_indices(geometry)
                if indices is not None:
                    indices_by_key[figure_key] = indices
                prepared.pop(ApiField.GEOMETRY, None)

            prepared_figures.append(prepared)

        return prepared_figures, figure_keys, indices_by_key

    def _download_bulk_from_entity_rows(
        self,
        dataset_id: int,
        mesh_ids: List[int],
        download_mesh_geometries: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[Dict]:
        if len(mesh_ids) == 0:
            return []

        project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
        obj_class_id_to_name = {
            obj_class.id: obj_class.name for obj_class in self._api.object_class.get_list(project_id)
        }
        tag_id_to_name = {tag.id: tag.name for tag in self._api.mesh.tag.get_list(project_id)}

        objects_by_mesh_id = {mesh_id: [] for mesh_id in mesh_ids}
        object_ids_by_mesh_id = {mesh_id: set() for mesh_id in mesh_ids}
        object_id_to_json = {}
        object_id_to_key = {}
        object_infos = self._api.mesh.object.get_list(dataset_id)
        for object_info in object_infos:
            object_key = uuid.uuid4()
            object_id_to_key[object_info.id] = object_key
            object_json = {
                KEY: object_key.hex,
                ApiField.ID: object_info.id,
                LabelJsonFields.OBJ_CLASS_NAME: obj_class_id_to_name.get(object_info.class_id),
                TAGS: self._convert_tag_rows_to_json(object_info.tags, tag_id_to_name),
            }
            if object_info.entity_id is not None:
                objects_by_mesh_id.setdefault(object_info.entity_id, []).append(object_json)
                object_ids_by_mesh_id.setdefault(object_info.entity_id, set()).add(object_info.id)
            object_id_to_json[object_info.id] = object_json

        figures_by_mesh_id = {mesh_id: [] for mesh_id in mesh_ids}
        mesh_geometry_refs = []
        raw_figures_by_mesh_id = self._api.mesh.figure.download(dataset_id, mesh_ids)
        for mesh_id, figure_infos in raw_figures_by_mesh_id.items():
            for figure_info in figure_infos:
                object_key = object_id_to_key.get(figure_info.object_id)
                if object_key is None:
                    object_key = uuid.uuid4()
                    object_id_to_key[figure_info.object_id] = object_key
                figure_json = {
                    KEY: uuid.uuid4().hex,
                    ApiField.ID: figure_info.id,
                    OBJECT_KEY: object_key.hex,
                    ApiField.GEOMETRY_TYPE: figure_info.geometry_type,
                    ApiField.GEOMETRY: figure_info.geometry,
                }
                if figure_info.tags:
                    figure_json[TAGS] = self._convert_tag_rows_to_json(
                        figure_info.tags, tag_id_to_name
                    )
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
                figures_by_mesh_id.setdefault(mesh_id, []).append(figure_json)

                object_json = object_id_to_json.get(figure_info.object_id)
                if object_json is not None and LabelJsonFields.OBJ_CLASS_NAME not in figure_json:
                    figure_json[LabelJsonFields.OBJ_CLASS_NAME] = object_json.get(
                        LabelJsonFields.OBJ_CLASS_NAME
                    )
                if (
                    object_json is not None
                    and figure_info.object_id not in object_ids_by_mesh_id.setdefault(mesh_id, set())
                ):
                    objects_by_mesh_id.setdefault(mesh_id, []).append(object_json)
                    object_ids_by_mesh_id[mesh_id].add(figure_info.object_id)

        if len(mesh_geometry_refs) != 0:
            figure_ids = [figure_id for figure_id, _ in mesh_geometry_refs]
            indices_batch = self._api.mesh.figure.download_indices_batch(figure_ids)
            for (_, figure_json), indices in zip(mesh_geometry_refs, indices_batch):
                figure_json[ApiField.GEOMETRY] = {INDICES: indices}

        annotations = []
        for mesh_id in mesh_ids:
            annotations.append(
                {
                    KEY: uuid.uuid4().hex,
                    MESH_ID: mesh_id,
                    TAGS: [],
                    OBJECTS: objects_by_mesh_id.get(mesh_id, []),
                    FIGURES: figures_by_mesh_id.get(mesh_id, []),
                }
            )
            if progress_cb is not None:
                self._update_progress(progress_cb, 1)
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
    def _uuid_from_value(value) -> Optional[uuid.UUID]:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        try:
            return uuid.UUID(str(value))
        except Exception:
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
