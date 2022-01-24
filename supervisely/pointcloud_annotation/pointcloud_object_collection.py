# coding: utf-8

from supervisely.video_annotation.video_object_collection import VideoObjectCollection
from supervisely.pointcloud_annotation.pointcloud_object import PointcloudObject


class PointcloudObjectCollection(VideoObjectCollection):
    '''
    Collection that stores PointcloudObject instances.
    '''
    item_type = PointcloudObject