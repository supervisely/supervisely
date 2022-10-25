# coding: utf-8
from __future__ import annotations
from typing import List, Dict, Optional, Iterator
from supervisely.video_annotation.video_object_collection import VideoObjectCollection
from supervisely.pointcloud_annotation.pointcloud_object import PointcloudObject
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.key_id_map import KeyIdMap

class PointcloudObjectCollection(VideoObjectCollection):
    '''
    Collection that stores PointcloudObject instances.
    '''
    item_type = PointcloudObject

    def __iter__(self) -> Iterator[PointcloudObject]:
        return next(self)

    @classmethod
    def from_json(
        cls, 
        data: List[Dict], 
        project_meta: ProjectMeta, 
        key_id_map: Optional[KeyIdMap]=None
    ) -> PointcloudObjectCollection:
        return super().from_json(data, project_meta, key_id_map=key_id_map)