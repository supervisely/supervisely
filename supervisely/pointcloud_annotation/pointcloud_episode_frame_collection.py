from __future__ import annotations
from typing import List, Dict, Optional, Any, Iterator

from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.pointcloud_annotation.pointcloud_episode_frame import PointcloudEpisodeFrame
from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

class PointcloudEpisodeFrameCollection(FrameCollection):
    item_type = PointcloudEpisodeFrame

    def get(self, key: str, default: Optional[Any]=None) -> PointcloudEpisodeFrame:
        return super().get(key, default)

    def __iter__(self) -> Iterator[PointcloudEpisodeFrame]:
        return next(self)

    @classmethod
    def from_json(
        cls, 
        data: List[Dict], 
        objects: PointcloudObjectCollection, 
        frames_count: Optional[int] = None,
        key_id_map: Optional[KeyIdMap] = None
    ) -> PointcloudEpisodeFrameCollection:
        return super().from_json(data, objects, frames_count=frames_count, key_id_map=key_id_map)

    @property
    def figures(self) -> List[PointcloudFigure]:
        return super().figures