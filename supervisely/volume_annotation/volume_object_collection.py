# coding: utf-8
from supervisely.video_annotation.video_object_collection import VideoObjectCollection
from supervisely.volume_annotation.volume_object import VolumeObject


class VolumeObjectCollection(VideoObjectCollection):
    item_type = VolumeObject
