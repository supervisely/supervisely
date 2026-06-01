import os
from typing import Dict, List, Optional

import supervisely.convert.pointcloud.pcd_semantic_labels.pcd_semantic_labels_helper as helpers
from supervisely import (
    ObjClass,
    PointcloudAnnotation,
    PointcloudFigure,
    PointcloudObject,
    ProjectMeta,
    logger,
)
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.io.fs import get_file_ext
from supervisely.pointcloud.pointcloud import validate_ext as validate_pcd_ext
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)


class PCDSemanticLabelsConverter(PointcloudConverter):
    """Imports PCD files with a numeric per-point labels field and class_mapping.json."""

    mapping_file_name = "class_mapping.json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._supports_links = True

    def __str__(self) -> str:
        return AvailablePointcloudConverters.PCD_SEMANTIC_LABELS

    def validate_format(self) -> bool:
        mapping_path = self._find_mapping_file()
        if mapping_path is None:
            return False

        class_mapping = helpers.read_class_mapping(mapping_path)
        pcd_paths = []
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                try:
                    validate_pcd_ext(get_file_ext(full_path))
                    pcd_paths.append(full_path)
                except Exception:
                    continue

        items = []
        meta = ProjectMeta()
        has_labeled_items = False
        for pcd_path in pcd_paths:
            indices_by_class = helpers.read_pcd_label_indices(pcd_path, class_mapping)
            if len(indices_by_class) > 0:
                has_labeled_items = True
            for class_name in indices_by_class.keys():
                if meta.get_obj_class(class_name) is None:
                    meta = meta.add_obj_class(ObjClass(class_name, Pointcloud))
            items.append(self.Item(pcd_path, ann_data=indices_by_class))

        if len(items) == 0 or not has_labeled_items:
            return False

        self._meta = meta
        self._items = items
        return True

    def to_supervisely(
        self,
        item: PointcloudConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> PointcloudAnnotation:
        """Convert PCD labels to Supervisely point_cloud figures."""
        if meta is None:
            meta = self._meta
        renamed_classes = renamed_classes if renamed_classes is not None else {}

        if not item.ann_data:
            return item.create_empty_annotation()

        objects = []
        figures = []
        indices_by_class: Dict[str, List[int]] = item.ann_data
        for class_name, indices in indices_by_class.items():
            class_name = renamed_classes.get(class_name, class_name)
            obj_class = meta.get_obj_class(class_name)
            if obj_class is None:
                obj_class = ObjClass(class_name, Pointcloud)
            pcd_obj = PointcloudObject(obj_class)
            objects.append(pcd_obj)
            figures.append(PointcloudFigure(pcd_obj, Pointcloud(indices=indices)))

        return PointcloudAnnotation(
            objects=PointcloudObjectCollection(objects),
            figures=figures,
        )

    def _find_mapping_file(self) -> Optional[str]:
        mapping_paths = []
        for root, _, files in os.walk(self._input_data):
            for file in files:
                if file == self.mapping_file_name:
                    mapping_paths.append(os.path.join(root, file))
        if len(mapping_paths) == 0:
            return None
        if len(mapping_paths) > 1:
            raise RuntimeError(
                f"Found multiple {self.mapping_file_name} files. "
                "PCD Semantic Labels format expects exactly one class mapping file."
            )
        return mapping_paths[0]
