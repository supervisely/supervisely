# coding: utf-8

# docs
from supervisely.pointcloud_annotation.pointcloud_episode_tag_collection import (
    PointcloudEpisodeTagCollection,
)
from supervisely.pointcloud_annotation.pointcloud_object import PointcloudObject


class PointcloudEpisodeObject(PointcloudObject):
    """Object tracked across point cloud episode frames (obj_class + tags). Uses VideoObject constructor. Immutable."""

    tag_collection_type = PointcloudEpisodeTagCollection
