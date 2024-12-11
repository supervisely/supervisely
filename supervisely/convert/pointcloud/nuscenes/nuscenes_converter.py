# supervisely/convert/pointcloud/nuscenes/nuscenes_converter.py
from typing import Dict, Optional, Union

from supervisely import PointcloudAnnotation
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.project.project_settings import LabelingInterface


class NuScenesConverter(PointcloudConverter):
    """Converter for nuScenes pointcloud format."""

    class Item(PointcloudConverter.Item):
        def __init__(
            self,
            item_path,
            ann_data: Optional[str] = None,
            related_images: Optional[list] = None,
            custom_data: Optional[dict] = None,
        ):
            super().__init__(item_path, ann_data, related_images, custom_data)
            self._type = "point_cloud"

    def __init__(
        self,
        input_data: str,
        labeling_interface: Optional[Union[LabelingInterface, str]],
        upload_as_links: bool,
        remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)
        self._type = "point_cloud"

    def __str__(self) -> str:
        return AvailablePointcloudConverters.NUSCENES
