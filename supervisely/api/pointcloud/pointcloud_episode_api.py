# coding: utf-8

# docs
from typing import Dict

from supervisely.api.module_api import ApiField
from supervisely.api.pointcloud.pointcloud_api import PointcloudApi
from supervisely.api.pointcloud.pointcloud_episode_annotation_api import PointcloudEpisodeAnnotationAPI


class PointcloudEpisodeApi(PointcloudApi):
    """
    API for working with :class:`PointcloudEpisode<supervisely.pointcloud_episodes.pointcloud_episodes>`.
    :class:`PointcloudEpisodeApi<PointcloudEpisodeApi>` object is immutable.
    Inherits from :class:`PointcloudApi<supervisely.api.pointcloud.PointcloudApi>`.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # You can connect to API directly
        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Or you can use API from environment
        os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
        os.environ['API_TOKEN'] = 'Your Supervisely API Token'
        api = sly.Api.from_env()

        pcd_epsodes_id = 19373295
        pcd_epsodes_info = api.pointcloud_episode.get_info_by_id(pcd_epsodes_id) # api usage example
    """

    def __init__(self, api):
        super().__init__(api)
        self.annotation = PointcloudEpisodeAnnotationAPI(api)

    def _convert_json_info(self, info: dict, skip_missing=True):
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        if res.meta is not None:
            return res._replace(frame=res.meta[ApiField.FRAME])
        else:
            raise RuntimeError('Error with point cloud meta or API version. Please, contact support')

    def get_frame_name_map(self, dataset_id: int) -> Dict:
        """
        Get a dictionary with frame_id and name of pointcloud by dataset id.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
        :type dataset_id: int
        :return: Dict with frame_id and name of pointcloud.
        :rtype: Dict

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 62664
            frame_to_name_map = api.pointcloud_episode.get_frame_name_map(dataset_id)
            print(frame_to_name_map)

            # Output:
            # {0: '001', 1: '002'}
        """

        pointclouds = self.get_list(dataset_id)

        frame_index_to_pcl_name = {}
        if len(pointclouds) > 0 and pointclouds[0].frame is None:
            pointclouds_names = sorted([x.name for x in pointclouds])
            for frame_index, pcl_name in enumerate(pointclouds_names):
                frame_index_to_pcl_name[frame_index] = pcl_name

        else:
            frame_index_to_pcl_name = {x.frame: x.name for x in pointclouds}

        return frame_index_to_pcl_name


