# coding: utf-8

from supervisely.api.pointcloud.pointcloud_object_api import PointcloudObjectApi
from supervisely.api.pointcloud.pointcloud_episode_tag_api import PointcloudEpisodeObjectTagApi


class PointcloudEpisodeObjectApi(PointcloudObjectApi):
    """
    API for working with point cloud episode objects.
    """

    def __init__(self, api):
        """
        :param api: :class:`~supervisely.api.api.Api` object to use for API connection.
        :type api: :class:`~supervisely.api.api.Api`
        """
        super().__init__(api)
        self.tag = PointcloudEpisodeObjectTagApi(api)
