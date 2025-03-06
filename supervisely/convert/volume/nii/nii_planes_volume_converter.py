import os
from collections import defaultdict
from pathlib import Path

from supervisely import ProjectMeta, logger
from supervisely.annotation.obj_class import ObjClass
from supervisely.convert.volume.nii import nii_volume_helper as helper
from supervisely.convert.volume.nii.nii_volume_converter import NiiConverter
from supervisely.convert.volume.volume_converter import VolumeConverter
from supervisely.geometry.mask_3d import Mask3D
from supervisely.io.fs import get_file_name
from supervisely.volume.volume import is_nifti_file
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely.volume_annotation.volume_object import VolumeObject


class NiiPlaneStructuredConverter(NiiConverter, VolumeConverter):
    """Convert NIfTI 3D volume file to Supervisely format.
    The NIfTI file should be structured as follows:
    - <prefix>_anatomic_<idx>.nii (or .nii.gz)
    - <prefix>_inference_<idx>.nii (or .nii.gz)
        where   <prefix> is one of the following: cor, sag, axl
                <idx> is the index of the volume (to match volumes with annotations)

    Supports .nii and .nii.gz files.

    Example:
        📂 .
        ├── 🩻 axl_anatomic_1.nii
        ├── 🩻 axl_inference_1.nii class 1 (may contain multiple instances of the same class)
        ├── 🩻 cor_anatomic_1.nii
        ├── 🩻 cor_inference_1.nii class 1
        ├── 🩻 sag_anatomic_1.nii
        ├── 🩻 sag_inference_1.nii class 1
        ├── 🩻 sag_inference_2.nii class 2
        └── 🩻 sag_inference_3.nii class 3
    """

    def validate_format(self) -> bool:
        # create Items
        converted_dir_name = "converted"

        volumes_dict = defaultdict(list)
        ann_dict = defaultdict(list)

        for root, _, files in os.walk(self._input_data):
            if converted_dir_name in root:
                continue
            for file in files:
                path = os.path.join(root, file)
                if is_nifti_file(path):
                    full_name = get_file_name(path)
                    if full_name.endswith(".nii"):
                        full_name = get_file_name(full_name)
                    prefix = full_name.split("_")[0]
                    if prefix not in helper.PlanePrefix.values():
                        continue
                    name = full_name.split("_")[1]
                    idx = 1 if len(name.split("_")) < 3 else int(name.split("_")[2])
                    if name in helper.LABEL_NAME or name[:-1] in helper.LABEL_NAME:
                        ann_dict[prefix].append(path)
                    else:
                        volumes_dict[prefix].append(path)

        self._items = []
        for prefix, paths in volumes_dict.items():
            if len(paths) == 1:
                item = self.Item(item_path=paths[0])
                item.ann_data = ann_dict.get(prefix)
                self._items.append(item)
            elif len(paths) > 1:
                logger.info(
                    f"Found {len(paths)} volumes with prefix {prefix}. Will try to match them by directories."
                )
                for path in paths:
                    item = self.Item(item_path=path)
                    possible_ann_paths = []
                    for ann_path in ann_dict.get(prefix):
                        if Path(ann_path).parent == Path(path).parent:
                            possible_ann_paths.append(ann_path)
                    item.ann_data = possible_ann_paths
                    self._items.append(item)
        self._meta = ProjectMeta()
        return self.items_count > 0

    def to_supervisely(
        self,
        item: VolumeConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> VolumeAnnotation:
        """Convert to Supervisely format."""

        try:
            objs = []
            spatial_figures = []
            for idx, ann_path in enumerate(item.ann_data, start=1):
                for mask, _ in helper.get_annotation_from_nii(ann_path):
                    class_name = f"Segment_{idx}"
                    class_name = renamed_classes.get(class_name, class_name)
                    obj_class = meta.get_obj_class(class_name)
                    if obj_class is None:
                        obj_class = ObjClass(class_name, Mask3D)
                        meta = meta.add_obj_class(obj_class)
                        self._meta_changed = True
                        self._meta = meta
                    obj = VolumeObject(obj_class, mask_3d=mask)
                    spatial_figures.append(obj.figure)
                    objs.append(obj)
            return VolumeAnnotation(item.volume_meta, objects=objs, spatial_figures=spatial_figures)
        except Exception as e:
            logger.warning(f"Failed to convert {item.path} to Supervisely format: {e}")
            return item.create_empty_annotation()
