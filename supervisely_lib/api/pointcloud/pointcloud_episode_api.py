# coding: utf-8
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.api.pointcloud.pointcloud_api import PointcloudApi
from supervisely_lib.api.pointcloud.pointcloud_episode_annotation_api import PointcloudEpisodeAnnotationAPI


class PointcloudEpisodeApi(PointcloudApi):
    def __init__(self, api):
        super().__init__(api)
        self.annotation = PointcloudEpisodeAnnotationAPI(api)

    def _convert_json_info(self, info: dict, skip_missing=True):
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        if res.meta is not None:
            return res._replace(frame=res.meta[ApiField.FRAME])
        else:
            raise RuntimeError('Error with point cloud meta or API version. Please, contact support')

    def get_frame_name_map(self, dataset_id):
        pointclouds = self.get_list(dataset_id)
        return {x.frame: x.name for x in pointclouds}


