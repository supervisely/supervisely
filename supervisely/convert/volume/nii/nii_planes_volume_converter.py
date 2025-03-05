import os
from collections import defaultdict

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
    - <prefix>_anatomic_<idx>.nii
    - <prefix>_inference_<idx>.nii
        where   <prefix> is one of the following: cor, sag, axl
                <idx> is the index of the volume (to match volumes with annotations)

    Example:
        📂 .
        ├── 🩻 axl_anatomic_1.nii
        ├── 🩻 axl_inference_1.nii
        ├── 🩻 cor_anatomic_1.nii
        ├── 🩻 cor_inference_1.nii
        ├── 🩻 sag_anatomic_1.nii
        └── 🩻 sag_inference_1.nii
    """

    def validate_format(self) -> bool:
        # create Items
        converted_dir_name = "converted"

        volumes_dict = defaultdict(lambda: {})
        ann_dict = defaultdict(lambda: {})

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
                    if name not in [helper.VOLUME_NAME, helper.LABEL_NAME]:
                        continue
                    idx = 1 if len(name.split("_")) < 3 else int(name.split("_")[2])
                    if name == helper.LABEL_NAME:
                        ann_dict[idx][prefix] = path
                    else:
                        volumes_dict[idx][prefix] = path

        self._items = []
        for idx, planes in volumes_dict.items():
            for prefix, path in planes.items():
                item = self.Item(item_path=path)
                item.ann_data = ann_dict.get(idx, {}).get(prefix)
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
            for mask, class_id in helper.get_annotation_from_nii(item.ann_data):
                class_name = f"Segment_{class_id}"
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
