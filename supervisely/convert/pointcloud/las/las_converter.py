import os
from typing import List

import supervisely.convert.pointcloud.las.las_helper as las_helper
from supervisely import PointcloudAnnotation, ProjectMeta, logger
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.io.fs import JUNK_FILES, get_file_ext


class LasConverter(PointcloudConverter):

    def __str__(self) -> str:
        return AvailablePointcloudConverters.LAS

    def validate_format(self) -> bool:
        las_list = []
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if file in JUNK_FILES:
                    continue
                elif ext in [".las", ".laz"]:
                    las_list.append(full_path)

        # create Items
        self._items = []

        # Warning about coordinate shift
        if len(las_list) > 0:
            logger.info(
                "⚠️ IMPORTANT: Coordinate shift will be applied to all LAS/LAZ files during conversion to PCD format. "
                "This is necessary to avoid floating-point precision issues and visual artifacts. "
                "The shift values (X, Y, Z offsets) will be logged for each file. "
                "If you need to convert annotations back to original LAS coordinates or use them with original LAS files, "
                "you MUST add these shift values back to the PCD/annotation coordinates. "
                "Check the logs for 'Applied coordinate shift' messages for each file."
            )

        for las_path in las_list:
            ext = get_file_ext(las_path)
            pcd_path = las_path.replace(ext, ".pcd")
            las_helper.las2pcd(las_path, pcd_path)
            if not os.path.exists(pcd_path):
                logger.warning(f"Failed to convert LAS/LAZ to PCD. Skipping: {las_path}")
                continue
            item = self.Item(pcd_path)
            self._items.append(item)
        return self.items_count > 0

    def to_supervisely(
        self,
        item: PointcloudConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> PointcloudAnnotation:
        """Convert to Supervisely format."""
        return item.create_empty_annotation()
