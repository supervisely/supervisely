from __future__ import annotations
import uuid
from typing import Optional, Dict, Union, Tuple
from supervisely.annotation.tag_meta import TagMeta
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_tag import VideoTag

class PointcloudEpisodeTag(VideoTag):
    @classmethod
    def from_json(cls, data: Dict, tag_meta_collection: TagMetaCollection, key_id_map: Optional[KeyIdMap] = None) -> PointcloudEpisodeTag:
        return super().from_json(data, tag_meta_collection, key_id_map=key_id_map)

    def __eq__(self, other: PointcloudEpisodeTag) -> bool:
        return super().__eq__(other)

    def clone(self, meta: Optional[TagMeta] = None, value: Optional[Union[str, int, float]] = None, frame_range: Optional[Tuple[int, int]] = None,
              key: Optional[uuid.UUID] = None, sly_id: Optional[int] = None, labeler_login: Optional[str] = None,
              updated_at: Optional[str] = None, created_at: Optional[str] = None) -> PointcloudEpisodeTag:
        return super().clone(
            meta=meta,
            value=value,
            frame_range=frame_range,
            key=key,
            sly_id=sly_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )