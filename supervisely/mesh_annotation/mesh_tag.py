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
        return MeshTag(
            meta=take_with_default(meta, self.meta),
            value=take_with_default(value, self.value),
            key=take_with_default(key, self.key()),
            sly_id=take_with_default(sly_id, self.sly_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at),
        )
