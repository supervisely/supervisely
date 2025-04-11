import os
from collections import defaultdict
from pathlib import Path

from supervisely import ProjectMeta, logger, Api
from supervisely.annotation.obj_class import ObjClass
from supervisely.convert.volume.nii import nii_volume_helper as helper
from supervisely.convert.volume.nii.nii_volume_converter import NiiConverter
from supervisely.convert.volume.volume_converter import VolumeConverter
from supervisely.geometry.mask_3d import Mask3D
from supervisely.io.fs import get_file_ext, get_file_name, list_files_recursively
from supervisely.volume.volume import is_nifti_file
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely.volume_annotation.volume_object import VolumeObject
from supervisely._utils import batched, is_development


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

    Additionally, if a TXT file with class color map is present, it will be used to
    create the classes with names and colors corresponding to the pixel values in the NIfTI files.
    The TXT file should be structured as follows:

    ```txt
    1 Femur 255 0 0
    2 Femoral cartilage 0 255 0
    3 Tibia 0 0 255
    4 Tibia cartilage 255 255 0
    5 Patella 0 255 255
    6 Patellar cartilage 255 0 255
    7 Miniscus 175 175 175
    ```
    where   1, 2, ... are the pixel values in the NIfTI files
            Femur, Femoral cartilage, ... are the names of the classes
            255, 0, 0, ... are the RGB colors of the classes
    The class name will be used to create the corresponding ObjClass in Supervisely.
    """

    class Item(VolumeConverter.BaseItem):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._is_semantic = False
            self.volume_meta = None

        @property
        def is_semantic(self) -> bool:
            return self._is_semantic

        @is_semantic.setter
        def is_semantic(self, value: bool):
            self._is_semantic = value

        def create_empty_annotation(self):
            return VolumeAnnotation(self.volume_meta)

    def validate_format(self) -> bool:
        # create Items
        converted_dir_name = "converted"

        volumes_dict = defaultdict(list)
        ann_dict = defaultdict(list)
        cls_color_map = None

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
                    if name in helper.LABEL_NAME or name[:-1] in helper.LABEL_NAME:
                        ann_dict[prefix].append(path)
                    else:
                        volumes_dict[prefix].append(path)
                ext = get_file_ext(path)
                if ext == ".txt":
                    cls_color_map = helper.read_cls_color_map(path)
                    if cls_color_map is None:
                        logger.warning(f"Failed to read class color map from {path}.")

        self._items = []
        for prefix, paths in volumes_dict.items():
            if len(paths) == 1:
                item = self.Item(item_path=paths[0])
                item.ann_data = ann_dict.get(prefix, [])
                item.is_semantic = len(item.ann_data) == 1
                if cls_color_map is not None:
                    item.custom_data["cls_color_map"] = cls_color_map
                self._items.append(item)
            elif len(paths) > 1:
                logger.info(
                    f"Found {len(paths)} volumes with prefix {prefix}. Will try to match them by directories."
                )
                for path in paths:
                    item = self.Item(item_path=path)
                    possible_ann_paths = []
                    for ann_path in ann_dict.get(prefix, []):
                        if Path(ann_path).parent == Path(path).parent:
                            possible_ann_paths.append(ann_path)
                    item.ann_data = possible_ann_paths
                    item.is_semantic = len(possible_ann_paths) == 1
                    if cls_color_map is not None:
                        item.custom_data["cls_color_map"] = cls_color_map
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
                for mask, pixel_id in helper.get_annotation_from_nii(ann_path):
                    class_id = pixel_id if item.is_semantic else idx
                    class_name = f"Segment_{class_id}"
                    color = None
                    if item.custom_data.get("cls_color_map") is not None:
                        class_info = item.custom_data["cls_color_map"].get(class_id)
                        if class_info is not None:
                            class_name, color = class_info
                    class_name = renamed_classes.get(class_name, class_name)
                    obj_class = meta.get_obj_class(class_name)
                    if obj_class is None:
                        obj_class = ObjClass(class_name, Mask3D, color)
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


class NiiPlaneStructuredAnnotationConverter(NiiConverter, VolumeConverter):
    """
    Upload NIfTI Annotations
    """

    class Item(VolumeConverter.BaseItem):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._is_semantic = False
            self.volume_meta = None

        @property
        def is_semantic(self) -> bool:
            return self._is_semantic

        @is_semantic.setter
        def is_semantic(self, value: bool):
            self._is_semantic = value

        def create_empty_annotation(self):
            return VolumeAnnotation(self.volume_meta)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._json_map = None

    def validate_format(self) -> bool:
        try:
            from nibabel import load, filebasedimages
        except ImportError:
            raise ImportError(
                "No module named nibabel. Please make sure that module is installed from pip and try again."
            )
        cls_color_map = None

        has_volumes = lambda x: x.split("_")[1] == helper.VOLUME_NAME if "_" in x else False
        if list_files_recursively(self._input_data, filter_fn=has_volumes):
            return False

        txts = list_files_recursively(self._input_data, [".txt"], None, True)
        cls_color_map = next(iter(txts), None)
        if cls_color_map is not None:
            cls_color_map = helper.read_cls_color_map(cls_color_map)

        jsons = list_files_recursively(self._input_data, [".json"], None, True)
        json_map = next(iter(jsons), None)
        if json_map is not None:
            self._json_map = helper.read_json_map(json_map)
            if self._json_map is None:
                logger.warning(f"Failed to read json map from {json_map}.")

        is_ann = lambda x: x.split("_")[1] in helper.LABEL_NAME if "_" in x else False
        for root, _, files in os.walk(self._input_data):
            for file in files:
                path = os.path.join(root, file)
                if is_ann(file):
                    prefix = get_file_name(path).split("_")[0]
                    if prefix not in helper.PlanePrefix.values():
                        continue
                    try:
                        nii = load(path)
                    except filebasedimages.ImageFileError:
                        continue
                    item = self.Item(item_path=None, ann_data=path)
                    item.set_shape(nii.shape)
                    if cls_color_map is not None:
                        item.custom_data["cls_color_map"] = cls_color_map
                    self._items.append(item)

        obj_classes = None
        if cls_color_map is not None:
            obj_classes = [ObjClass(name, Mask3D, color) for name, color in cls_color_map.values()]

        self._meta = ProjectMeta(obj_classes=obj_classes)
        return len(self._items) > 0

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
            ann_path = item.ann_data
            for mask, pixel_id in helper.get_annotation_from_nii(ann_path):
                class_id = pixel_id if item.is_semantic else 1
                class_name = f"Segment_{class_id}"
                color = None
                if item.custom_data.get("cls_color_map") is not None:
                    class_info = item.custom_data["cls_color_map"].get(class_id)
                    if class_info is not None:
                        class_name, color = class_info
                class_name = renamed_classes.get(class_name, class_name)
                obj_class = meta.get_obj_class(class_name)
                if obj_class is None:
                    obj_class = ObjClass(class_name, Mask3D, color)
                    meta = meta.add_obj_class(obj_class)
                    self._meta_changed = True
                    self._meta = meta
                obj = VolumeObject(obj_class, mask_3d=mask)
                spatial_figures.append(obj.figure)
                objs.append(obj)
            return VolumeAnnotation(item.volume_meta, objects=objs, spatial_figures=spatial_figures)
        except Exception as e:
            logger.warning(f"Failed to convert {item.ann_data} to Supervisely format: {e}")
            return item.create_empty_annotation()

    def upload_dataset(
        self, api: Api, dataset_id: int, batch_size: int = 50, log_progress=True
    ) -> None:
        meta, renamed_classes, _ = self.merge_metas_with_conflicts(api, dataset_id)

        matcher = helper.AnnotationMatcher(self._items, dataset_id)
        if self._json_map is not None:
            try:
                matched_dict = matcher.match_from_json(api, self._json_map)
            except Exception as e:
                logger.error(f"Failed to match annotations from a json map: {e}")
                matched_dict = {}
        else:
            matcher.get_volumes(api)
            matched_dict = matcher.match_items()
            if len(matched_dict) != len(self._items):
                extra = {
                    "items count": len(self._items),
                    "matched count": len(matched_dict),
                    "unmatched count": len(self._items) - len(matched_dict),
                }
                logger.warning(
                    "Not all items were matched with volumes. Some items may be skipped.",
                    extra=extra,
                )
        if len(matched_dict) == 0:
            raise RuntimeError(
                "No items were matched with volumes. Please check the input data and try again."
            )

        if log_progress:
            progress, progress_cb = self.get_progress(
                len(matched_dict), "Uploading volumes annotations..."
            )
        else:
            progress_cb = None

        for item, volume in matched_dict.items():
            item.volume_meta = volume.meta
            ann = self.to_supervisely(item, meta, renamed_classes, None)
            if self._meta_changed:
                meta, renamed_classes, _ = self.merge_metas_with_conflicts(api, dataset_id)
                self._meta_changed = False
            api.volume.annotation.append(volume.id, ann, volume_info=volume)
            progress_cb(1) if log_progress else None

        if log_progress:
            if is_development():
                progress.close()
            logger.info(f"Successfully uploaded {len(matched_dict)} annotations.")
