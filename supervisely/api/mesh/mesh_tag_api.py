# coding: utf-8
from __future__ import annotations

from typing import List, Optional, Union

from supervisely.api.entity_annotation.tag_api import TagApi
from supervisely.api.module_api import ApiField
from supervisely.mesh_annotation.constants import KEY


class MeshTagApi(TagApi):
    """API for tags attached directly to mesh entities."""

    _entity_id_field = ApiField.ENTITY_ID
    _method_bulk_add = "tags.entities.bulk.add"

    @staticmethod
    def _clean_entity_tag_json(tag_json: dict) -> dict:
        """Return a copy of the tag JSON without the ``name`` and key fields."""
        tag_json = dict(tag_json)
        tag_json.pop(ApiField.NAME, None)
        tag_json.pop(KEY, None)
        return tag_json

    def add(
        self,
        tag_meta_id: int,
        mesh_id: int,
        value: Optional[Union[str, int, float]] = None,
        project_id: Optional[int] = None,
    ) -> int:
        """
        Add a tag to a mesh entity and return the created tag ID.

        :param tag_meta_id: ID of the tag meta (tag definition) to assign.
        :type tag_meta_id: int
        :param mesh_id: ID of the mesh entity to attach the tag to.
        :type mesh_id: int
        :param value: Tag value. Required for tags whose meta defines a value type.
        :type value: str or int or float, optional
        :param project_id: ID of the project. If not provided, it is resolved from *mesh_id*.
        :type project_id: int, optional
        :returns: ID of the created tag.
        :rtype: int

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            tag_id = api.mesh.tag.add(tag_meta_id=123, mesh_id=456, value="car")
        """
        if project_id is None:
            project_id = self._api.mesh.get_info_by_id(mesh_id).project_id
        request_body = {
            ApiField.PROJECT_ID: project_id,
            ApiField.TAGS: [
                {
                    ApiField.TAG_ID: tag_meta_id,
                    ApiField.ENTITY_ID: mesh_id,
                }
            ],
        }
        if value is not None:
            request_body[ApiField.TAGS][0][ApiField.VALUE] = value
        response = self._api.post("tags.entities.bulk.add", request_body)
        return response.json()[0][ApiField.ID]

    def append_to_entity(
        self,
        entity_id: int,
        project_id: int,
        tags,
    ) -> List[int]:
        """
        Add multiple tags to a mesh entity and return their created IDs.

        :param entity_id: ID of the mesh entity to attach the tags to.
        :type entity_id: int
        :param project_id: ID of the project the entity belongs to.
        :type project_id: int
        :param tags: Tags to append to the entity.
        :type tags: :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>`
        :returns: List of created tag IDs, ordered like the input tags.
        :rtype: List[int]
        """
        if len(tags) == 0:
            return []

        tags_json, _ = self._tags_to_json(tags, project_id=project_id)
        for tag_json in tags_json:
            tag_json[ApiField.ENTITY_ID] = entity_id

        response = self._api.post(
            "tags.entities.bulk.add",
            {
                ApiField.PROJECT_ID: project_id,
                ApiField.TAGS: [self._clean_entity_tag_json(tag_json) for tag_json in tags_json],
            },
        )
        return [obj[ApiField.ID] for obj in response.json()]


    def remove(self, tag_id: int) -> None:
        """Remove a tag from a mesh entity."""
        self._api.post("tags.entities.remove", {ApiField.ID: tag_id})

    def update_value(self, tag_id: int, value: Union[str, int]) -> None:
        """Update the value of a tag on a mesh entity."""
        self._api.post("tags.entities.update-value", {ApiField.ID: tag_id, ApiField.VALUE: value})
