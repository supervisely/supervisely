from __future__ import annotations
from typing import Optional, List, Dict

from supervisely._utils import take_with_default
from supervisely.video_annotation.frame import Frame
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

class PointcloudEpisodeFrame(Frame):
    figure_type = PointcloudFigure

    def __init__(self, index: int, figures: Optional[List[PointcloudFigure]]=None):
        super().__init__(index, figures)

    @classmethod
    def from_json(
        cls, 
        data: Dict, 
        objects: PointcloudObjectCollection, 
        frames_count: Optional[int]=None, 
        key_id_map: Optional[KeyIdMap]=None
    ) -> PointcloudEpisodeFrame:
        return super().from_json(data, objects, frames_count, key_id_map)

    def clone(self, index: Optional[int] = None, figures: Optional[List[PointcloudFigure]] = None) -> PointcloudEpisodeFrame:
        return super().clone(index, figures)