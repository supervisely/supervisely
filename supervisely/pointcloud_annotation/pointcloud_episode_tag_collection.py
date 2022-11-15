from __future__ import annotations
from typing import List, Dict, Optional, Any, Iterator
from supervisely.pointcloud_annotation.pointcloud_tag_collection import PointcloudTagCollection
from supervisely.pointcloud_annotation.pointcloud_episode_tag import PointcloudEpisodeTag
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.annotation.tag_meta import TagMeta
from supervisely.video_annotation.key_id_map import KeyIdMap

class PointcloudEpisodeTagCollection(PointcloudTagCollection):
    item_type = PointcloudEpisodeTag

    def __iter__(self) -> Iterator[PointcloudEpisodeTag]:
        return next(self)

    @classmethod
    def from_api_response(
        cls, 
        data: List[Dict], 
        tag_meta_collection: TagMetaCollection, 
        id_to_tagmeta: Optional[Dict[int, TagMeta]] = None
    ) -> PointcloudEpisodeTagCollection:
        return super().from_api_response(data, tag_meta_collection, id_to_tagmeta=id_to_tagmeta)

    def get_by_name(self, tag_name: str) -> List[PointcloudEpisodeTag]:
        return super().get_by_name(tag_name)

    def get_single_by_name(self, tag_name: str) -> PointcloudEpisodeTag:
        return super().get_single_by_name(tag_name)

    @classmethod
    def from_json(
        cls, 
        data: List[Dict], 
        tag_meta_collection: TagMetaCollection, 
        key_id_map: Optional[KeyIdMap] = None,
    ) -> PointcloudEpisodeTagCollection:
        return super().from_json(data, tag_meta_collection, key_id_map=key_id_map)