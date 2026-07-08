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
        """Initialize a mesh annotation as a flat list of labels and entity tags.

        :param labels: Mesh labels (geometry-backed objects) of the annotation.
        :type labels: Optional[List[:class:`~supervisely.mesh_annotation.mesh_label.MeshLabel`]]
        :param tags: Entity-level tags attached to the mesh.
        :type tags: Optional[:class:`~supervisely.mesh_annotation.mesh_tag_collection.MeshTagCollection`]
        :param description: Free-text description of the annotation.
        :type description: Optional[str]
        :param key: Unique identifier of the annotation. Generated automatically if not provided.
        :type key: Optional[uuid.UUID]
        """
        self._description = take_with_default(description, "")
        self._tags = take_with_default(tags, MeshTagCollection())
        self._labels = take_with_default(labels, [])
        self._key = take_with_default(key, uuid.uuid4())

    @property
    def description(self) -> str:
        """Free-text description of the annotation.

        :rtype: str
        """
        return self._description

    @property
    def tags(self) -> MeshTagCollection:
        """Entity-level tags attached to the mesh (returned as a copy).

        :rtype: :class:`~supervisely.mesh_annotation.mesh_tag_collection.MeshTagCollection`
        """
        return self._tags.clone()

    @property
    def labels(self) -> List[MeshLabel]:
        """Mesh labels of the annotation (returned as a shallow copy of the list).

        :rtype: List[:class:`~supervisely.mesh_annotation.mesh_label.MeshLabel`]
        """
        return self._labels.copy()

    def key(self) -> uuid.UUID:
        """Return the unique identifier of the annotation.

        :returns: Annotation key.
        :rtype: uuid.UUID
        """
        return self._key

    def to_json(self) -> Dict:
        """Serialize the annotation to a JSON-serializable dict.

        :returns: Dict with description, key, tags and labels of the annotation.
        :rtype: dict
        """
        return {
            DESCRIPTION: self.description,
            KEY: self.key().hex,
            TAGS: self.tags.to_json(),
            LABELS: [label.to_json() for label in self.labels],
        }

    @classmethod
    def from_json(cls, data: Dict, project_meta: ProjectMeta) -> "MeshAnnotation":
        """Deserialize a mesh annotation from a JSON dict.

        :param data: Mesh annotation in JSON format.
        :type data: dict
        :param project_meta: Project meta used to resolve object classes and tag metas.
        :type project_meta: :class:`~supervisely.project.project_meta.ProjectMeta`
        :returns: Deserialized mesh annotation.
        :rtype: :class:`~supervisely.mesh_annotation.mesh_annotation.MeshAnnotation`
        :raises RuntimeError: If the JSON uses the unsupported legacy ``objects``/``figures`` schema.
        """
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
        """Load and deserialize a mesh annotation from a JSON file.

        :param path: Path to the JSON file.
        :type path: str
        :param project_meta: Project meta used to resolve object classes and tag metas.
        :type project_meta: :class:`~supervisely.project.project_meta.ProjectMeta`
        :returns: Deserialized mesh annotation.
        :rtype: :class:`~supervisely.mesh_annotation.mesh_annotation.MeshAnnotation`
        """
        with open(path) as fin:
            data = json.load(fin)
        return cls.from_json(data, project_meta)

    def clone(
        self,
        labels: Optional[List[MeshLabel]] = None,
        tags: Optional[MeshTagCollection] = None,
        description: Optional[str] = None,
    ) -> "MeshAnnotation":
        """Return a copy of the annotation with the given fields overridden.

        :param labels: New mesh labels. Keeps current labels if not provided.
        :type labels: Optional[List[:class:`~supervisely.mesh_annotation.mesh_label.MeshLabel`]]
        :param tags: New entity tags. Keeps current tags if not provided.
        :type tags: Optional[:class:`~supervisely.mesh_annotation.mesh_tag_collection.MeshTagCollection`]
        :param description: New description. Keeps current description if not provided.
        :type description: Optional[str]
        :returns: New mesh annotation with the same key and the overridden fields.
        :rtype: :class:`~supervisely.mesh_annotation.mesh_annotation.MeshAnnotation`
        """
        return MeshAnnotation(
            labels=take_with_default(labels, self.labels),
            tags=take_with_default(tags, self.tags),
            description=take_with_default(description, self.description),
            key=self.key(),
        )
