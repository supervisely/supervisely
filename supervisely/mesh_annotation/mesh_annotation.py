# coding: utf-8
from __future__ import annotations

import json
import uuid
from copy import deepcopy
from typing import Dict, List, Optional

from supervisely._utils import take_with_default
from supervisely.mesh_annotation.constants import DESCRIPTION, FIGURES, KEY, MESH_ID, OBJECTS, TAGS
from supervisely.mesh_annotation.mesh_figure import MeshFigure
from supervisely.mesh_annotation.mesh_object_collection import MeshObjectCollection
from supervisely.mesh_annotation.mesh_tag_collection import MeshTagCollection
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_annotation import VideoAnnotation


class MeshAnnotation(VideoAnnotation):
    """Annotation for a single mesh entity."""

    def __init__(
        self,
        objects: Optional[MeshObjectCollection] = None,
        figures: Optional[List[MeshFigure]] = None,
        tags: Optional[MeshTagCollection] = None,
        description: Optional[str] = "",
        key: Optional[uuid.UUID] = None,
    ):
        self._description = description
        self._tags = take_with_default(tags, MeshTagCollection())
        self._objects = take_with_default(objects, MeshObjectCollection())
        self._figures = take_with_default(figures, [])
        self._key = take_with_default(key, uuid.uuid4())

    @property
    def img_size(self):
        raise NotImplementedError("Not supported for meshes")

    @property
    def frames_count(self):
        raise NotImplementedError("Not supported for meshes")

    @property
    def frames(self):
        raise NotImplementedError("Not supported for meshes")

    @property
    def tags(self) -> MeshTagCollection:
        return super().tags

    @property
    def objects(self) -> MeshObjectCollection:
        return super().objects

    @property
    def figures(self) -> List[MeshFigure]:
        return deepcopy(self._figures)

    def get_objects_from_figures(self) -> MeshObjectCollection:
        ann_objects = {}
        for figure in self.figures:
            if figure.parent_object.key() not in ann_objects:
                ann_objects[figure.parent_object.key()] = figure.parent_object
        return MeshObjectCollection(ann_objects.values())

    def validate_figures_bounds(self):
        raise NotImplementedError("Not supported for meshes")

    def to_json(self, key_id_map: Optional[KeyIdMap] = None) -> Dict:
        res_json = {
            DESCRIPTION: self.description,
            KEY: self.key().hex,
            TAGS: self.tags.to_json(key_id_map),
            OBJECTS: self.objects.to_json(key_id_map),
            FIGURES: [figure.to_json(key_id_map) for figure in self.figures],
        }

        if key_id_map is not None:
            mesh_id = key_id_map.get_video_id(self.key())
            if mesh_id is not None:
                res_json[MESH_ID] = mesh_id

        return res_json

    @classmethod
    def from_json(
        cls, data: Dict, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> "MeshAnnotation":
        try:
            item_key = uuid.UUID(data[KEY])
        except Exception:
            item_key = uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_video(item_key, data.get(MESH_ID, None))
        description = data.get(DESCRIPTION, "")
        tags = MeshTagCollection.from_json(data.get(TAGS, []), project_meta.tag_metas, key_id_map)
        objects = MeshObjectCollection.from_json(data.get(OBJECTS, []), project_meta, key_id_map)

        figures = []
        for figure_json in data.get(FIGURES, []):
            figure = MeshFigure.from_json(figure_json, objects, None, key_id_map)
            figures.append(figure)

        return cls(objects=objects, figures=figures, tags=tags, description=description, key=item_key)

    @classmethod
    def load_json_file(
        cls, path: str, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> "MeshAnnotation":
        with open(path) as fin:
            data = json.load(fin)
        return cls.from_json(data, project_meta, key_id_map)

    def clone(
        self,
        objects: Optional[MeshObjectCollection] = None,
        figures: Optional[List[MeshFigure]] = None,
        tags: Optional[MeshTagCollection] = None,
        description: Optional[str] = None,
    ) -> "MeshAnnotation":
        return MeshAnnotation(
            objects=take_with_default(objects, self.objects),
            figures=take_with_default(figures, self.figures),
            tags=take_with_default(tags, self.tags),
            description=take_with_default(description, self.description),
        )
