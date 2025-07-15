import os
from typing import List

import supervisely.convert.volume.sly.sly_volume_helper as sly_volume_helper
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely import ProjectMeta, logger
from supervisely.project.volume_project import VolumeProject, VolumeDataset
from supervisely.convert.base_converter import AvailableVolumeConverters
from supervisely.convert.volume.volume_converter import VolumeConverter
from supervisely.io.fs import JUNK_FILES, get_file_ext, get_file_name
from supervisely.io.json import load_json_file
from supervisely.volume.volume import is_valid_ext as validate_volume_ext


class SLYVolumeConverter(VolumeConverter):

    def __str__(self) -> str:
        return AvailableVolumeConverters.SLY

    @property
    def ann_ext(self) -> str:
        return ".json"

    @property
    def key_file_ext(self) -> str:
        return ".json"

    def generate_meta_from_annotation(
        self, ann_path: str, meta: ProjectMeta
    ) -> ProjectMeta:
        meta = sly_volume_helper.get_meta_from_annotation(ann_path, meta)
        return meta

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta) -> bool:
        try:
            ann_json = load_json_file(ann_path)
            for key in sly_volume_helper.SLY_VOLUME_ANN_KEYS:
                value = ann_json.get(key)
                if value is None:
                    return False
            ann = VolumeAnnotation.from_json(ann_json, meta)  # , KeyIdMap())
            return True
        except Exception as e:
            logger.warning(f"Failed to validate annotation: {repr(e)}")
            return False

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            self._meta = ProjectMeta.from_json(load_json_file(key_file_path))
            return True
        except Exception:
            return False

    def validate_format(self) -> bool:
        if self.read_sly_project(self._input_data):
            return True

        detected_ann_cnt = 0
        vol_list, stl_dict, ann_dict, mask_dict = [], {}, {}, {}
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                # if file == "key_id_map.json":
                #     key_id_map_json = load_json_file(full_path)
                #     self._key_id_map = KeyIdMap.from_dict(key_id_map_json)
                if file == "meta.json":
                    is_valid = self.validate_key_file(full_path)
                    if is_valid:
                        if self._meta is not None:
                            meta_json = load_json_file(full_path)
                            try:
                                self._meta = self._meta.merge(
                                    ProjectMeta.from_json(meta_json)
                                )
                            except Exception as e:
                                logger.warning(f"Failed to merge meta: {repr(e)}")
                        continue

                elif file in JUNK_FILES:  # add better check
                    continue
                elif ext == ".stl":
                    stl_dir = os.path.dirname(full_path)
                    stl_dirname = os.path.basename(stl_dir)
                    stl_dict[stl_dirname] = full_path
                elif ext in self.ann_ext:
                    ann_dict[file] = full_path
                elif ext == ".dcm":
                    return False
                elif root.endswith(".nrrd") and os.path.isdir(root):
                    mask_dir = os.path.dirname(full_path)
                    mask_dirname = os.path.basename(mask_dir)
                    mask_dict[mask_dirname] = mask_dir
                else:
                    try:
                        is_valid = validate_volume_ext(ext)
                        if is_valid:
                            vol_list.append(full_path)
                    except:
                        continue

        if self._meta is not None:
            meta = self._meta
        else:
            meta = ProjectMeta()

        # create Items
        self._items = []
        for vol_path in vol_list:
            item = self.Item(vol_path)
            ann_name = f"{item.name}.json"
            if ann_name in ann_dict:
                ann_path = ann_dict[ann_name]
                if self._meta is None:
                    meta = self.generate_meta_from_annotation(ann_path, meta)
                is_valid = self.validate_ann_file(ann_path, meta)
                if is_valid:
                    item.ann_data = ann_path
                    detected_ann_cnt += 1
            item.mask_dir = mask_dict.get(item.name)
            item.interpolation_dir = stl_dict.get(item.name)
            self._items.append(item)
        self._meta = meta
        return detected_ann_cnt > 0

    def to_supervisely(
        self,
        item: VolumeConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> VolumeAnnotation:
        """Convert to Supervisely format."""
        if meta is None:
            meta = self._meta

        if item.ann_data is None:
            return item.create_empty_annotation()

        try:
            ann_json = load_json_file(item.ann_data)
            if renamed_classes or renamed_tags:
                ann_json = sly_volume_helper.rename_in_json(ann_json, renamed_classes, renamed_tags)
            return VolumeAnnotation.from_json(ann_json, meta)  # , KeyIdMap())
        except Exception as e:
            logger.warning(f"Failed to read annotation: {repr(e)}")
            return item.create_empty_annotation()

    def read_sly_project(self, input_data: str) -> bool:
        try:
            project_fs = VolumeProject.read_single(input_data)
            self._meta = project_fs.meta
            self._items = []
            
            for dataset_fs in project_fs.datasets:
                dataset_fs: VolumeDataset

                for item_name in dataset_fs:
                    img_path, ann_path = dataset_fs.get_item_paths(item_name)
                    item = self.Item(
                        item_path=img_path,
                        ann_data=ann_path,
                        shape=None,
                        interpolation_dir=dataset_fs.get_interpolation_dir(item_name),
                        mask_dir=dataset_fs.get_mask_dir(item_name),
                    )
                    self._items.append(item)
            return True

        except Exception as e:
            logger.info(f"Failed to read Supervisely project: {repr(e)}")
            return False
