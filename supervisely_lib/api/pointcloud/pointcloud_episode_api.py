# coding: utf-8
from supervisely_lib.api.pointcloud.pointcloud_api import PointcloudApi
from supervisely_lib.api.pointcloud.pointcloud_episode_annotation_api import PointcloudEpisodeAnnotationAPI

class PointcloudEpisodeApi(PointcloudApi):
    def __init__(self, api):
        '''
        :param api: Api class object
        '''
        super().__init__(api)
        self.annotation = PointcloudEpisodeAnnotationAPI(api)

    def get_frame_name_map(self, dataset_id):
        pointclouds = self.get_list(dataset_id)
        return {x.frame: x.name for x in pointclouds}
