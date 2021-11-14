# coding: utf-8

from supervisely_lib.video_annotation.video_object_collection import VideoObjectCollection
from supervisely_lib.volume_annotation.volume_object import VolumeObject


class VolumeObjectCollection(VideoObjectCollection):
    item_type = VolumeObject

    def __str__(self):
        return 'Objects:\n' + super(VolumeObjectCollection, self).__str__()
