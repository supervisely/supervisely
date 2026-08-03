# coding: utf-8
from __future__ import annotations

import uuid
from typing import Optional, Union

from supervisely._utils import take_with_default
from supervisely.annotation.tag_meta import TagMeta
from supervisely.pointcloud_annotation.pointcloud_tag import PointcloudTag


class MeshTag(PointcloudTag):
    """Tag attached to a mesh annotation."""

    def clone(
        self,
        meta: Optional[TagMeta] = None,
        value: Optional[Union[str, int, float]] = None,
        key: Optional[uuid.UUID] = None,
        sly_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> "MeshTag":
        """Return a copy of the tag with the given fields overridden.

        Any argument left as ``None`` keeps the current value of the tag.

        :param meta: New tag meta.
        :type meta: Optional[:class:`~supervisely.annotation.tag_meta.TagMeta`]
        :param value: New tag value.
        :type value: Optional[Union[str, int, float]]
        :param key: New unique identifier.
        :type key: Optional[uuid.UUID]
        :param sly_id: New tag ID in Supervisely.
        :type sly_id: Optional[int]
        :param labeler_login: New labeler login.
        :type labeler_login: Optional[str]
        :param updated_at: New update timestamp.
        :type updated_at: Optional[str]
        :param created_at: New creation timestamp.
        :type created_at: Optional[str]
        :returns: New mesh tag with the overridden fields.
        :rtype: :class:`~supervisely.mesh_annotation.mesh_tag.MeshTag`
        """
        return MeshTag(
            meta=take_with_default(meta, self.meta),
            value=take_with_default(value, self.value),
            key=take_with_default(key, self.key()),
            sly_id=take_with_default(sly_id, self.sly_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at),
        )
