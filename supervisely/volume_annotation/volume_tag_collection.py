# coding: utf-8

from supervisely.video_annotation.video_tag_collection import VideoTagCollection
from supervisely.volume_annotation.volume_tag import VolumeTag


class VolumeTagCollection(VideoTagCollection):
    item_type = VolumeTag
