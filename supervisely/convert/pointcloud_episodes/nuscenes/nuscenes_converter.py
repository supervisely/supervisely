# supervisely/convert/pointcloud_episodes/nuscenes/nuscenes_converter.py
from typing import Dict, Optional, Union

from supervisely import PointcloudEpisodeAnnotation
from supervisely.convert.base_converter import AvailablePointcloudEpisodesConverters
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import (
    PointcloudEpisodeConverter,
)
from supervisely.project.project_settings import LabelingInterface


class NuScenesEpisodesConverter(PointcloudEpisodeConverter):
    """Converter for nuScenes pointcloud episodes format."""

    class Item(PointcloudEpisodeConverter.Item):
        def __init__(
            self,
            item_path,
            frame_number: int,
            ann_data: Optional[str] = None,
            related_images: Optional[list] = None,
            custom_data: Optional[dict] = None,
        ):
            super().__init__(item_path, frame_number, ann_data, related_images, custom_data)
            self._type = "point_cloud_episode"

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

    def __str__(self) -> str:
        return AvailablePointcloudEpisodesConverters.NUSCENES
