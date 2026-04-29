# coding: utf-8
from __future__ import annotations

import uuid
from typing import Dict, Optional

from supervisely.annotation.tag import TagJsonFields
from supervisely.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely.api.module_api import ApiField
from supervisely.geometry.constants import CREATED_AT, LABELER_LOGIN, UPDATED_AT
from supervisely.mesh_annotation.mesh_annotation import MeshAnnotation
from supervisely.mesh_annotation.mesh_object_collection import MeshObjectCollection
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.constants import ID, KEY, OBJECT_ID
from supervisely.video_annotation.key_id_map import KeyIdMap


class MeshAnnotationAPI(EntityAnnotationAPI):
    """API for downloading and appending mesh annotations."""

    def append(
        self,
        mesh_id: int,
        ann: MeshAnnotation,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> None:
        info = self._api.mesh.get_info_by_id(mesh_id)

        new_objects = []
        for mesh_object in ann.objects:
            if key_id_map is not None and key_id_map.get_object_id(mesh_object.key()) is not None:
                continue
            new_objects.append(mesh_object)

        self._append(
            self._api.mesh.tag,
            self._api.mesh.object,
            self._api.mesh.figure,
            info.project_id,
            info.dataset_id,
            mesh_id,
            ann.tags,
            MeshObjectCollection(new_objects),
            ann.figures,
            key_id_map,
        )

    def download(
        self,
        mesh_id: int,
        project_meta: Optional[ProjectMeta] = None,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> Dict:
        if key_id_map is None:
            key_id_map = KeyIdMap()

        mesh_info_json = self._api.mesh._get_json_info_by_id(
            mesh_id, fields=self._api.mesh.default_fields() + [ApiField.TAGS]
        )
        mesh_info = self._api.mesh._convert_json_info(mesh_info_json)

        if project_meta is None:
            project_meta = ProjectMeta.from_json(self._api.project.get_meta(mesh_info.project_id))

        mesh_key = uuid.uuid4()
        key_id_map.add_video(mesh_key, mesh_id)

        tags = self._tags_from_api(mesh_info_json.get(ApiField.TAGS, []), project_meta, key_id_map)
        objects = self._download_objects(mesh_info.dataset_id, mesh_id, project_meta, key_id_map)
        figures = self._download_figures(mesh_info.dataset_id, mesh_id, objects, key_id_map)

        ann = MeshAnnotation(
            objects=objects,
            figures=figures,
            tags=tags,
            description=mesh_info.description or "",
            key=mesh_key,
        )
        return ann.to_json(key_id_map)

    def _download_objects(
        self,
        dataset_id: int,
        mesh_id: int,
        project_meta: ProjectMeta,
        key_id_map: KeyIdMap,
    ) -> MeshObjectCollection:
        from supervisely.mesh_annotation.mesh_object import MeshObject
        filters = [
            {ApiField.FIELD: ApiField.ENTITY_ID, ApiField.OPERATOR: "=", ApiField.VALUE: mesh_id}
        ]
        object_infos = self._api.mesh.object.get_list(dataset_id, filters=filters)

        objects = []
        for object_info in object_infos:
            obj_class = project_meta.get_obj_class_by_id(object_info.class_id)
            if obj_class is None:
                raise RuntimeError(
                    f"Object class with id={object_info.class_id} was not found in project meta."
                )

            tags = self._tags_from_api(object_info.tags or [], project_meta, key_id_map)
            object_key = uuid.uuid4()
            key_id_map.add_object(object_key, object_info.id)
            objects.append(
                MeshObject(
                    obj_class=obj_class,
                    tags=tags,
                    key=object_key,
                    class_id=object_info.class_id,
                    labeler_login=getattr(object_info, "labeler_login", None),
                    updated_at=object_info.updated_at,
                    created_at=object_info.created_at,
                )
            )

        return MeshObjectCollection(objects)

    def _download_figures(
        self,
        dataset_id: int,
        mesh_id: int,
        objects: MeshObjectCollection,
        key_id_map: KeyIdMap,
    ):
        from supervisely.mesh_annotation.mesh_figure import MeshFigure

        figures_by_entity = self._api.mesh.figure.download(dataset_id, [mesh_id])
        figures = []
        for figure_info in figures_by_entity.get(mesh_id, []):
            figure_json = figure_info.to_json()
            figure_json[OBJECT_ID] = figure_info.object_id
            if KEY not in figure_json:
                figure_json[KEY] = uuid.uuid4().hex
            figure = MeshFigure.from_json(figure_json, objects, None, key_id_map)
            figures.append(figure)
        return figures

    def _tags_from_api(
        self,
        tags_json,
        project_meta: ProjectMeta,
        key_id_map: Optional[KeyIdMap] = None,
    ):
        from supervisely.mesh_annotation.mesh_tag import MeshTag
        from supervisely.mesh_annotation.mesh_tag_collection import MeshTagCollection

        tags = []
        for tag_info in tags_json or []:
            tag_meta = None
            tag_meta_id = tag_info.get(ApiField.TAG_ID)
            if tag_meta_id is not None:
                tag_meta = project_meta.tag_metas.get_by_id(tag_meta_id)
            if tag_meta is None and tag_info.get(ApiField.NAME) is not None:
                tag_meta = project_meta.tag_metas.get(tag_info[ApiField.NAME])
            if tag_meta is None:
                raise RuntimeError(f"TagMeta with id={tag_meta_id!r} was not found in project meta.")

            tag_json = {
                TagJsonFields.TAG_NAME: tag_meta.name,
                KEY: uuid.uuid4().hex,
            }
            if tag_info.get(ApiField.VALUE) is not None:
                tag_json[TagJsonFields.VALUE] = tag_info[ApiField.VALUE]
            if tag_info.get(ApiField.ID) is not None:
                tag_json[ID] = tag_info[ApiField.ID]
            if tag_info.get(LABELER_LOGIN) is not None:
                tag_json[LABELER_LOGIN] = tag_info[LABELER_LOGIN]
            if tag_info.get(UPDATED_AT) is not None:
                tag_json[UPDATED_AT] = tag_info[UPDATED_AT]
            if tag_info.get(CREATED_AT) is not None:
                tag_json[CREATED_AT] = tag_info[CREATED_AT]
            tags.append(MeshTag.from_json(tag_json, project_meta.tag_metas, key_id_map))

        return MeshTagCollection(tags)
