import os
from typing import List

import supervisely.convert.pointcloud.las.las_helper as las_helper
from supervisely import PointcloudAnnotation, ProjectMeta, logger
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.io.fs import JUNK_FILES, get_file_ext


class LasConverter(PointcloudConverter):
    """Imports LAS/LAZ point cloud files (converts to PCD internally) into Supervisely point cloud project."""

    def __str__(self) -> str:
        return AvailablePointcloudConverters.LAS

    def validate_format(self) -> bool:
        # Deprecated: LAS/LAZ files are now natively supported by Supervisely.
        # Raw LAS/LAZ files are handled by the base PointcloudConverter;
        # LAS/LAZ files inside a SLY project are handled by SlyPointcloudConverter.
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
