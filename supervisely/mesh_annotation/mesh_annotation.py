# coding: utf-8
from __future__ import annotations

import json
import uuid
from typing import Dict, List, Optional

from supervisely._utils import take_with_default
from supervisely.mesh_annotation.constants import DESCRIPTION, FIGURES, KEY, LABELS, OBJECTS, TAGS
from supervisely.mesh_annotation.mesh_label import MeshLabel
from supervisely.mesh_annotation.mesh_tag_collection import MeshTagCollection
from supervisely.project.project_meta import ProjectMeta


class MeshAnnotation:
    """Annotation for a single mesh entity."""

    def __init__(
        self,
        labels: Optional[List[MeshLabel]] = None,
        tags: Optional[MeshTagCollection] = None,
        description: Optional[str] = "",
        key: Optional[uuid.UUID] = None,
    ):
        self._description = take_with_default(description, "")
        self._tags = take_with_default(tags, MeshTagCollection())
        self._labels = take_with_default(labels, [])
        self._key = take_with_default(key, uuid.uuid4())

    @property
    def description(self) -> str:
        return self._description

    @property
    def tags(self) -> MeshTagCollection:
        return self._tags.clone()

    @property
    def labels(self) -> List[MeshLabel]:
        return self._labels.copy()

    def key(self) -> uuid.UUID:
        return self._key

    def to_json(self) -> Dict:
        return {
            DESCRIPTION: self.description,
            KEY: self.key().hex,
            TAGS: self.tags.to_json(),
            LABELS: [label.to_json() for label in self.labels],
        }

    @classmethod
    def from_json(cls, data: Dict, project_meta: ProjectMeta) -> "MeshAnnotation":
        if OBJECTS in data or FIGURES in data:
            raise RuntimeError(
                "Legacy mesh annotation JSON with 'objects'/'figures' is not supported. "
                "Use the 'labels' mesh annotation schema."
            )

        try:
            item_key = uuid.UUID(data[KEY])
        except Exception:
            item_key = uuid.uuid4()

        description = data.get(DESCRIPTION, "")
        tags = MeshTagCollection.from_json(data.get(TAGS, []), project_meta.tag_metas)
        labels = [
            MeshLabel.from_json(label_json, project_meta)
            for label_json in data.get(LABELS, [])
        ]

        return cls(labels=labels, tags=tags, description=description, key=item_key)

    @classmethod
    def load_json_file(cls, path: str, project_meta: ProjectMeta) -> "MeshAnnotation":
        with open(path) as fin:
            data = json.load(fin)
        return cls.from_json(data, project_meta)

    def clone(
        self,
        labels: Optional[List[MeshLabel]] = None,
        tags: Optional[MeshTagCollection] = None,
        description: Optional[str] = None,
    ) -> "MeshAnnotation":
        return MeshAnnotation(
            labels=take_with_default(labels, self.labels),
            tags=take_with_default(tags, self.tags),
            description=take_with_default(description, self.description),
            key=self.key(),
        )
