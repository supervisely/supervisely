import os
from pathlib import Path

from supervisely import ProjectMeta, generate_free_name, logger
from supervisely._utils import batched, is_development
from supervisely.annotation.obj_class import ObjClass
from supervisely.api.api import Api
from supervisely.convert.base_converter import AvailableVolumeConverters
from supervisely.convert.volume.nii import nii_volume_helper as helper
from supervisely.convert.volume.volume_converter import VolumeConverter
from supervisely.geometry.mask_3d import Mask3D
from supervisely.io.fs import (
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    list_files,
)
from supervisely.task.progress import tqdm_sly
from supervisely.volume.volume import is_nifti_file, read_nrrd_serie_volume_np
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely.volume_annotation.volume_object import VolumeObject


class NiiConverter(VolumeConverter):
    """
    Convert NIfTI 3D volume file to Supervisely format.
    Supports .nii and .nii.gz files.

    The NIfTI file should be structured as follows:
    - <volume_name>.nii
    - <volume_name>/
        - <cls_name_1>.nii
        - <cls_name_2>.nii
        - ...
    - ...

    where   <volume_name> is the name of the volume
            If the volume has annotations, they should be in the corresponding directory
                with the same name as the volume (without extension)
            <cls_name> is the name of the annotation class
                <cls_name>.nii:
                    - represent objects of the single class
                    - should be unique for the current volume (e.g. tumor.nii.gz, lung.nii.gz)
                    - can contain multiple objects of the class (each object should be represented by a different value in the mask)

    Example:
    📂 .
    ├── 📂 CTChest
    │   ├── 🩻 lung.nii.gz
    │   └── 🩻 tumor.nii.gz
    ├── 🩻 CTChest.nii.gz
    └── 🩻 Spine.nii.gz
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._supports_links = True
        self._meta_changed = False

    def __str__(self) -> str:
        return AvailableVolumeConverters.NII

    def validate_format(self) -> bool:
        # create Items
        converted_dir_name = "converted"
        # nrrds_dict = {}
        nifti_dict = {}
        nifti_dirs = {}

        planes_detected = {p: False for p in helper.PlanePrefix.values()}

        for root, _, files in os.walk(self._input_data):
            dir_name = os.path.basename(root)
            nifti_dirs[dir_name] = root
            if converted_dir_name in root:
                continue
            for file in files:
                path = os.path.join(root, file)
                if is_nifti_file(path):  # is nifti
                    name = get_file_name(path)
                    if name.endswith(".nii"):
                        name = get_file_name(name)
                    nifti_dict[name] = path
                    for plane in planes_detected.keys():
                        if plane in name:
                            planes_detected[plane] = True

        if any(planes_detected.values()):
            return False

        self._items = []
        skip_files = []
        for name, nrrd_path in nifti_dict.items():
            if name in nifti_dirs:
                item = self.Item(item_path=nrrd_path)
                ann_dir = nifti_dirs[name]
                item.ann_data = list_files(ann_dir, [".nii", ".nii.gz", ".gz"], None, True)
                self._items.append(item)
                skip_files.extend(item.ann_data)
                skip_files.append(nrrd_path)

        for name, nrrd_path in nifti_dict.items():
            if nrrd_path in skip_files:
                continue
            item = self.Item(item_path=nrrd_path)
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
            for ann_path in item.ann_data:
                ann_name = get_file_name(ann_path)
                if ann_name.endswith(".nii"):
                    ann_name = get_file_name(ann_name)

                ann_name = renamed_classes.get(ann_name, ann_name)
                for mask, pixel_value in helper.get_annotation_from_nii(ann_path):
                    obj_class = meta.get_obj_class(ann_name)
                    if obj_class is None:
                        obj_class = ObjClass(
                            ann_name, Mask3D, description=f"{helper.MASK_PIXEL_VALUE}{pixel_value}"
                        )
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

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 1,
        log_progress=True,
    ):
        """Upload converted data to Supervisely"""

        meta, renamed_classes, _ = self.merge_metas_with_conflicts(api, dataset_id)

        existing_names = set([vol.name for vol in api.volume.get_list(dataset_id)])

        if log_progress:
            progress, progress_cb = self.get_progress(
                self.items_count, "Converting and uploading volumes..."
            )
        else:
            progress_cb = None

        converted_dir_name = "converted"
        converted_dir = os.path.join(self._input_data, converted_dir_name)

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []

            for item in batch:
                if self._upload_as_links:
                    remote_path = self.remote_files_map.get(item.path)
                    if remote_path is not None:
                        item.custom_data["remote_path"] = remote_path

                item.path = helper.nifti_to_nrrd(item.path, converted_dir)
                ext = get_file_ext(item.path)
                if ext.lower() != ext:
                    new_volume_path = Path(item.path).with_suffix(ext.lower()).as_posix()
                    os.rename(item.path, new_volume_path)
                    item.path = new_volume_path
                item.name = get_file_name_with_ext(item.path)
                item.name = generate_free_name(
                    existing_names, item.name, with_ext=True, extend_used_names=True
                )
                item_names.append(item.name)
                item_paths.append(item.path)

                # upload volume
                volume_np, volume_meta = read_nrrd_serie_volume_np(item.path)
                progress_nrrd = tqdm_sly(
                    desc=f"Uploading volume '{item.name}'",
                    total=sum(volume_np.shape) + 1,
                    leave=True if progress_cb is None else False,
                    position=1,
                )
                if isinstance(item.custom_data, dict) and "remote_path" in item.custom_data:
                    volume_meta["remote_path"] = item.custom_data["remote_path"]
                api.volume.upload_np(dataset_id, item.name, volume_np, volume_meta, progress_nrrd)
                info = api.volume.get_info_by_name(dataset_id, item.name)
                item.volume_meta = info.meta

                # create and upload annotation
                if item.ann_data is not None:
                    ann = self.to_supervisely(item, meta, renamed_classes, None)

                    if self._meta_changed:
                        meta, renamed_classes, _ = self.merge_metas_with_conflicts(api, dataset_id)
                        self._meta_changed = False
                    api.volume.annotation.append(info.id, ann)

            if log_progress:
                progress_cb(len(batch))

        if log_progress:
            if is_development():
                progress.close()
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")
