import os
from typing import List

import supervisely.convert.pointcloud.ply.ply_helper as ply_helper
from supervisely import PointcloudAnnotation, ProjectMeta, logger
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.io.fs import JUNK_FILES, get_file_ext, get_file_name


class PlyConverter(PointcloudConverter):
    """Imports PLY point cloud files (converts to PCD) with optional JSON annotations and RGB images."""

    def __str__(self) -> str:
        return AvailablePointcloudConverters.PLY

    @property
    def ann_ext(self) -> str:
        return ".json"

    def validate_format(self) -> bool:
        # Deprecated: PLY files are now natively supported by Supervisely.
        # Raw PLY files are handled by the base PointcloudConverter;
        # PLY files inside a SLY project are handled by SlyPointcloudConverter.
        return False

    def to_supervisely(
        self,
        item: PointcloudConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> PointcloudAnnotation:
        """Convert to Supervisely format."""
        return item.create_empty_annotation()