# coding: utf-8

# docs
from typing import Dict

from supervisely.api.module_api import ApiField
from supervisely.api.pointcloud.pointcloud_api import PointcloudApi
from supervisely.api.pointcloud.pointcloud_episode_annotation_api import PointcloudEpisodeAnnotationAPI


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

    def get_frame_name_map(self, dataset_id: int) -> Dict:
        pointclouds = self.get_list(dataset_id)

        frame_index_to_pcl_name = {}
        if len(pointclouds) > 0 and pointclouds[0].frame is None:
            pointclouds_names = sorted([x.name for x in pointclouds])
            for frame_index, pcl_name in enumerate(pointclouds_names):
                frame_index_to_pcl_name[frame_index] = pcl_name

        else:
            frame_index_to_pcl_name = {x.frame: x.name for x in pointclouds}

        return frame_index_to_pcl_name


