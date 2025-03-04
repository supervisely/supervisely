import os
from pathlib import Path

import magic

from supervisely import ProjectMeta, generate_free_name, logger
from supervisely._utils import batched, is_development
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
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
from supervisely.volume.volume import is_nifti_file
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely.volume_annotation.volume_object import VolumeObject


class NiiConverter(VolumeConverter):

    def __str__(self) -> str:
        return AvailableVolumeConverters.NII

    def validate_format(self) -> bool:
        # create Items
        converted_dir_name = "converted"
        # nrrds_dict = {}
        nifti_dict = {}
        nifti_dirs = {}
        for root, _, files in os.walk(self._input_data):
            dir_name = os.path.basename(root)
            nifti_dirs[dir_name] = root
            if converted_dir_name in root:
                continue
            for file in files:
                path = os.path.join(root, file)
                mime = magic.from_file(path, mime=True)
                if mime == "application/gzip" or mime == "application/octet-stream":
                    if is_nifti_file(path):  # is nifti
                        name = get_file_name(path)
                        if name.endswith(".nii"):
                            name = get_file_name(name)
                        nifti_dict[name] = path

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

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 1,
        log_progress=True,
    ):
        """Upload converted data to Supervisely"""

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        existing_names = set([vol.name for vol in api.volume.get_list(dataset_id)])

        if log_progress:
            progress, progress_cb = self.get_progress(
                self.items_count, "Converting and uploading volumes..."
            )
        else:
            progress_cb = None

        converted_dir_name = "converted"
        converted_dir = os.path.join(self._input_data, converted_dir_name)
        meta_changed = False

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []

            for item in batch:
                # nii_path = item.path
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

                volume_info = api.volume.upload_nrrd_serie_path(
                    dataset_id, name=item.name, path=item.path
                )

                if isinstance(item.ann_data, list) and len(item.ann_data) > 0:
                    objs = []
                    spatial_figures = []
                    for ann_path in item.ann_data:
                        ann_name = get_file_name(ann_path)
                        if ann_name.endswith(".nii"):
                            ann_name = get_file_name(ann_name)
                        for mask, _ in helper.get_annotation_from_nii(ann_path):
                            obj_class = meta.get_obj_class(ann_name)
                            if obj_class is None:
                                obj_class = ObjClass(ann_name, Mask3D)
                                meta = meta.add_obj_class(obj_class)
                                meta_changed = True
                            obj = VolumeObject(obj_class, mask_3d=mask)
                            spatial_figures.append(obj.figure)
                            objs.append(obj)
                    ann = VolumeAnnotation(
                        volume_info.meta, objects=objs, spatial_figures=spatial_figures
                    )

                    if meta_changed:
                        self._meta = meta
                        _, _, _ = self.merge_metas_with_conflicts(api, dataset_id)

                    api.volume.annotation.append(volume_info.id, ann)

            if log_progress:
                progress_cb(len(batch))

        if log_progress:
            if is_development():
                progress.close()
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")
