import imghdr
import os
from typing import List

import supervisely.convert.pointcloud.ply.ply_helper as ply_helper
from supervisely import PointcloudAnnotation, ProjectMeta, logger
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.io.fs import JUNK_FILES, get_file_ext, get_file_name


class PlyConverter(PointcloudConverter):

    def __str__(self) -> str:
        return AvailablePointcloudConverters.PLY

    @property
    def ann_ext(self) -> str:
        return ".json"

    def validate_format(self) -> bool:
        ply_list = []
        ann_dict = {}
        rimg_dict = {}
        used_img_ext = []
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if file in JUNK_FILES:
                    continue
                elif ext == ".ply":
                    ply_list.append(full_path)
                elif ext ==  self.ann_ext:
                    ann_dict[file] = full_path
                elif imghdr.what(full_path):
                    rimg_dict[file] = full_path
                    if ext not in used_img_ext:
                        used_img_ext.append(ext)

                    
                

        # create Items
        self._items = []
        for ply_path in ply_list:
            pcd_path = ply_path.replace(".ply", ".pcd")
            ply_helper.ply2pcd(ply_path, pcd_path)
            if not os.path.exists(pcd_path):
                logger.warn(f"Failed to convert PLY to PCD. Skipping: {ply_path}")
                continue
            item = self.Item(pcd_path)
            for ext in used_img_ext:
                rimg_name = f"{item.name}{ext}"
                if not rimg_name in rimg_dict:
                    rimg_name = f"{get_file_name(item.name)}{ext}"
                if rimg_name in rimg_dict:
                    rimg_path = rimg_dict[rimg_name]
                    rimg_ann_name = f"{rimg_name}.json"
                    if rimg_ann_name in ann_dict:
                        rimg_ann_path = ann_dict[rimg_ann_name]
                        item.set_related_images((rimg_path, rimg_ann_path))
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
