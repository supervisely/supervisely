# coding: utf-8

from supervisely_lib.collection.str_enum import StrEnum


class ProjectType(StrEnum):
    IMAGES = 'images'
    VIDEOS = 'videos'
    VOLUMES = 'volumes'
    POINT_CLOUDS = 'point_clouds'