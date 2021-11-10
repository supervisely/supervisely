# coding: utf-8
from supervisely_lib.api.pointcloud.pointcloud_api import PointcloudApi
from supervisely_lib.api.pointcloud.pointcloud_episode_annotation_api import PointcloudEpisodeAnnotationAPI
from supervisely_lib.api.pointcloud.pointcloud_object_api import PointcloudObjectApi
from supervisely_lib.api.pointcloud.pointcloud_figure_api import PointcloudFigureApi
from supervisely_lib.api.pointcloud.pointcloud_tag_api import PointcloudTagApi


class PointcloudEpisodeApi(PointcloudApi):
    def __init__(self, api):
        '''
        :param api: Api class object
        '''
        super().__init__(api)
        self.annotation = PointcloudEpisodeAnnotationAPI(api)
        self.object = PointcloudObjectApi(api)
        self.figure = PointcloudFigureApi(api)
        self.tag = PointcloudTagApi(api)
