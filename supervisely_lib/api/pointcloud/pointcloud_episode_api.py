# coding: utf-8
from supervisely_lib.api.pointcloud.pointcloud_api import PointcloudApi
from supervisely_lib.api.pointcloud.pointcloud_episode_annotation_api import PointcloudEpisodeAnnotationAPI


class PointcloudEpisodeApi(PointcloudApi):
    def __init__(self, api):
        super().__init__(api)
        self.annotation = PointcloudEpisodeAnnotationAPI(api)

    def get_frame_name_map(self, dataset_id):
        pointclouds = self.get_list(dataset_id)
        try:
            frame_name_map = {x.meta['frame']: x.name for x in pointclouds}
            test_value = list(frame_name_map.keys())[0]
            if test_value is not None and type(test_value) is int:
                return frame_name_map
            else:
                raise ValueError(f'Incorrect value of meta. Got: {type(test_value)}:{test_value}, Expected: int')
        except AttributeError:
            raise AttributeError("Pointcloud has no meta field")
        except TypeError:
            raise TypeError('Meta is empty. Please upload pointclouds with Meta params')
        except KeyError:
            raise KeyError('No frame provided in meta. Please upload point cloud with passing "frame" to meta')

