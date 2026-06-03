# coding: utf-8
from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Dict, List, Optional, Union
from uuid import UUID

from supervisely._utils import take_with_default
from supervisely.annotation.json_geometries_map import GET_GEOMETRY_FROM_STR
from supervisely.annotation.label import LabelJsonFields, LabelingStatus
from supervisely.annotation.obj_class import ObjClass
from supervisely.api.module_api import ApiField
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.constants import (
    CLASS_ID,
    CREATED_AT,
    GEOMETRY_SHAPE,
    GEOMETRY_TYPE,
    LABELER_LOGIN,
    UPDATED_AT,
)
from supervisely.geometry.geometry import Geometry
from supervisely.mesh_annotation.constants import KEY, TAGS
from supervisely.mesh_annotation.mesh_tag import MeshTag
from supervisely.mesh_annotation.mesh_tag_collection import MeshTagCollection
from supervisely.project.project_meta import ProjectMeta


class MeshLabel:
    """Single geometry-backed label in a mesh annotation."""

    def __init__(
        self,
        geometry: Geometry,
        obj_class: ObjClass,
        tags: Optional[Union[MeshTagCollection, List[MeshTag]]] = None,
        description: Optional[str] = "",
        key: Optional[UUID] = None,
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
        custom_data: Optional[Dict] = None,
        priority: Optional[int] = None,
        status: Optional[LabelingStatus] = None,
    ):
        self._geometry = geometry
        self._obj_class = obj_class
        self._tags = take_with_default(tags, MeshTagCollection())
        if isinstance(self._tags, list):
            self._tags = MeshTagCollection(self._tags)
        self._description = take_with_default(description, "")
        self._key = take_with_default(key, uuid.uuid4())
        self._sly_id = sly_id
        self._class_id = class_id
        self.labeler_login = labeler_login
        self.updated_at = updated_at
        self.created_at = created_at
        self._custom_data = deepcopy(take_with_default(custom_data, {}))
        self._priority = priority
        self._status = take_with_default(status, LabelingStatus.MANUAL)
        self._nn_created, self._nn_updated = LabelingStatus.to_flags(self._status)
        self._validate_geometry_type()
        self._validate_geometry()

    def _validate_geometry_type(self) -> None:
        if self.obj_class.geometry_type is AnyGeometry:
            return
        if type(self.geometry) is not self.obj_class.geometry_type:
            raise TypeError(
                "Geometry type {!r} does not match object class geometry type {!r}.".format(
                    type(self.geometry), self.obj_class.geometry_type
                )
            )

    def _validate_geometry(self) -> None:
        self.geometry.validate(
            self.obj_class.geometry_type.geometry_name(),
            self.obj_class.geometry_config,
        )

    def _add_creation_info(self, data: Dict) -> None:
        if self.labeler_login is not None:
            data[LABELER_LOGIN] = self.labeler_login
        if self.updated_at is not None:
            data[UPDATED_AT] = self.updated_at
        if self.created_at is not None:
            data[CREATED_AT] = self.created_at

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def obj_class(self) -> ObjClass:
        return self._obj_class

    @property
    def tags(self) -> MeshTagCollection:
        return self._tags.clone()

    @property
    def description(self) -> str:
        return self._description

    @property
    def custom_data(self) -> Dict:
        return deepcopy(self._custom_data)

    @property
    def priority(self) -> Optional[int]:
        return self._priority

    @property
    def sly_id(self) -> Optional[int]:
        return self._sly_id

    @property
    def class_id(self) -> Optional[int]:
        return self._class_id

    @property
    def status(self) -> LabelingStatus:
        return self._status

    def key(self) -> uuid.UUID:
        return self._key

    def to_json(self) -> Dict:
        geometry_json = deepcopy(self.geometry.to_json())
        geometry_json.pop(GEOMETRY_TYPE, None)
        geometry_json.pop(GEOMETRY_SHAPE, None)

        data_json = {
            KEY: self.key().hex,
            LabelJsonFields.OBJ_CLASS_NAME: self.obj_class.name,
            LabelJsonFields.DESCRIPTION: self.description,
            TAGS: self.tags.to_json(),
            ApiField.GEOMETRY_TYPE: self.geometry.geometry_name(),
            ApiField.GEOMETRY: geometry_json,
            ApiField.NN_CREATED: self._nn_created,
            ApiField.NN_UPDATED: self._nn_updated,
        }

        class_id = self.class_id if self.class_id is not None else self.obj_class.sly_id
        if class_id is not None:
            data_json[CLASS_ID] = class_id
        if self.sly_id is not None:
            data_json[LabelJsonFields.ID] = self.sly_id
        if self.priority is not None:
            data_json[ApiField.PRIORITY] = self.priority
        if self.custom_data:
            data_json[ApiField.CUSTOM_DATA] = self.custom_data

        self._add_creation_info(data_json)
        return data_json

    @classmethod
    def from_json(
        cls,
        data: Dict,
        project_meta: ProjectMeta,
    ) -> "MeshLabel":
        obj_class_name = data[LabelJsonFields.OBJ_CLASS_NAME]
        obj_class = project_meta.get_obj_class(obj_class_name)
        if obj_class is None:
            raise RuntimeError(
                f"Failed to deserialize a MeshLabel from JSON: class name "
                f"{obj_class_name!r} was not found in the given project meta."
            )

        geometry_type = data[ApiField.GEOMETRY_TYPE]
        geometry_json = data.get(ApiField.GEOMETRY)
        if not isinstance(geometry_json, dict):
            raise RuntimeError("Mesh label can not be deserialized without geometry.")

        if obj_class.geometry_type is AnyGeometry:
            geometry_cls = GET_GEOMETRY_FROM_STR(geometry_type)
        else:
            geometry_cls = obj_class.geometry_type
        geometry = geometry_cls.from_json(geometry_json)

        key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()
        label_id = data.get(LabelJsonFields.ID)

        nn_created = data.get(ApiField.NN_CREATED, False)
        nn_updated = data.get(ApiField.NN_UPDATED, False)

        return cls(
            geometry=geometry,
            obj_class=obj_class,
            tags=MeshTagCollection.from_json(data.get(TAGS, []), project_meta.tag_metas),
            description=data.get(LabelJsonFields.DESCRIPTION, ""),
            key=key,
            sly_id=label_id,
            class_id=data.get(CLASS_ID, data.get(LabelJsonFields.OBJ_CLASS_ID)),
            labeler_login=data.get(LABELER_LOGIN),
            updated_at=data.get(UPDATED_AT),
            created_at=data.get(CREATED_AT),
            custom_data=data.get(ApiField.CUSTOM_DATA),
            priority=data.get(ApiField.PRIORITY),
            status=LabelingStatus.from_flags(nn_created, nn_updated),
        )

    def clone(
        self,
        geometry: Optional[Geometry] = None,
        obj_class: Optional[ObjClass] = None,
        tags: Optional[Union[MeshTagCollection, List[MeshTag]]] = None,
        description: Optional[str] = None,
        key: Optional[UUID] = None,
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
        custom_data: Optional[Dict] = None,
        priority: Optional[int] = None,
        status: Optional[LabelingStatus] = None,
    ) -> "MeshLabel":
        return MeshLabel(
            geometry=take_with_default(geometry, self.geometry),
            obj_class=take_with_default(obj_class, self.obj_class),
            tags=take_with_default(tags, self.tags),
            description=take_with_default(description, self.description),
            key=take_with_default(key, self.key()),
            sly_id=take_with_default(sly_id, self.sly_id),
            class_id=take_with_default(class_id, self.class_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at),
            custom_data=take_with_default(custom_data, self.custom_data),
            priority=take_with_default(priority, self.priority),
            status=take_with_default(status, self.status),
        )
