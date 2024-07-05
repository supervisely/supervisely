from typing import Dict, Optional, Union

from supervisely.convert.base_converter import AvailablePointcloudEpisodesConverters
from supervisely.convert.pointcloud.bag.bag_converter import BagConverter
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import PointcloudEpisodeConverter
from supervisely.pointcloud_annotation.pointcloud_episode_annotation import PointcloudEpisodeAnnotation
from supervisely.project.project_settings import LabelingInterface


class BagEpisodesConverter(BagConverter, PointcloudEpisodeConverter):
    """Converter for ROS bag files containing point cloud data to use for point cloud episodes."""

    class Item(BagConverter.Item):
        def __init__(
            self,
            item_path,
            frame_number: int,
            ann_data: str = None,
            related_images: list = None,
            custom_data: dict = None,
        ):
            super().__init__(item_path, ann_data, related_images, custom_data)
            self._frame_number = frame_number

        def create_empty_annotation(self) -> PointcloudEpisodeAnnotation:
            return PointcloudEpisodeAnnotation()

    def __init__(
            self,
            input_data: str,
            labeling_interface: Optional[Union[LabelingInterface, str]],
            upload_as_links: bool,
            remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)

        self._type = "point_cloud_episode"
        self._is_pcd_episode = True

    def __str__(self) -> str:
        return AvailablePointcloudEpisodesConverters.BAG
