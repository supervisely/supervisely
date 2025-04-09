import imghdr
import os
from typing import List, Optional, Set, Tuple
import numpy as np

import supervisely.convert.pointcloud.sly.sly_pointcloud_helper as helpers
from supervisely import (
    Api,
    PointcloudAnnotation,
    batched,
    generate_free_name,
    is_development,
    logger,
)
from supervisely.api.module_api import ApiField
from supervisely.convert.base_converter import BaseConverter
from supervisely.io.fs import get_file_ext, list_files_recursively
from supervisely.io.json import load_json_file
from supervisely.pointcloud.pointcloud import ALLOWED_POINTCLOUD_EXTENSIONS
from supervisely.pointcloud.pointcloud import validate_ext as validate_pcd_ext


class PointcloudConverter(BaseConverter):
    allowed_exts = ALLOWED_POINTCLOUD_EXTENSIONS
    modality = "pointclouds"

    class Item(BaseConverter.BaseItem):
        def __init__(
            self,
            item_path,
            ann_data=None,
            related_images: Optional[list] = None,
            custom_data: Optional[dict] = None,
        ):
            self._name: str = None
            self._path = item_path
            self._ann_data = ann_data
            self._type = "point_cloud"
            self._related_images = related_images if related_images is not None else []
            self._custom_data = custom_data if custom_data is not None else {}

        def create_empty_annotation(self) -> PointcloudAnnotation:
            return PointcloudAnnotation()

        def set_related_images(self, related_images: Tuple[str, str]) -> None:
            self._related_images.append(related_images)

    @property
    def format(self):
        return self._converter.format

    @property
    def ann_ext(self):
        return None

    @property
    def key_file_ext(self):
        return None

    @staticmethod
    def validate_ann_file(ann_path, meta=None):
        return False

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 1,
        log_progress=True,
    ):
        """Upload converted data to Supervisely"""

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        existing_names = set([pcd.name for pcd in api.pointcloud.get_list(dataset_id)])

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading pointclouds...")
        else:
            progress_cb = None

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            anns = []
            for item in batch:
                item.name = generate_free_name(
                    existing_names, item.name, with_ext=True, extend_used_names=True
                )
                item_names.append(item.name)
                item_paths.append(item.path)

                ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
                anns.append(ann)

            pcd_infos = api.pointcloud.upload_paths(
                dataset_id,
                item_names,
                item_paths,
            )
            pcd_ids = [pcd_info.id for pcd_info in pcd_infos]

            for pcd_id, ann in zip(pcd_ids, anns):
                if ann is not None:
                    api.pointcloud.annotation.append(pcd_id, ann)

                rimg_infos = []
                camera_names = []
                for img_ind, (img_path, rimg_ann_path) in enumerate(item._related_images):
                    meta_json = load_json_file(rimg_ann_path)
                    try:
                        if ApiField.META not in meta_json:
                            raise ValueError("Related image meta not found in json file.")
                        if ApiField.NAME not in meta_json:
                            raise ValueError("Related image name not found in json file.")
                        img = api.pointcloud.upload_related_image(img_path)
                        if "deviceId" not in meta_json[ApiField.META].keys():
                            camera_names.append(f"CAM_{str(img_ind).zfill(2)}")
                        else:
                            camera_names.append(meta_json[ApiField.META]["deviceId"])
                        rimg_infos.append(
                            {
                                ApiField.ENTITY_ID: pcd_id,
                                ApiField.NAME: meta_json[ApiField.NAME],
                                ApiField.HASH: img,
                                ApiField.META: meta_json[ApiField.META],
                            }
                        )
                        api.pointcloud.add_related_images(rimg_infos, camera_names)
                    except Exception as e:
                        logger.warn(
                            f"Failed to upload related image or add it to pointcloud: {repr(e)}"
                        )
                        continue

            if log_progress:
                progress_cb(len(batch))

        if log_progress:
            if is_development():
                progress.close()
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")

    def _convert_bin_to_pcd(self, bin_file, save_filepath):
        import open3d as o3d  # pylint: disable=import-error

        b = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
        points = b[:, 0:3]
        intensity = b[:, 3]
        ring_index = b[:, 4]
        intensity_fake_rgb = np.zeros((intensity.shape[0], 3))
        intensity_fake_rgb[:, 0] = (
            intensity  # red The intensity measures the reflectivity of the objects
        )
        intensity_fake_rgb[:, 1] = (
            ring_index  # green ring index is the index of the laser ranging from 0 to 31
        )

        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pc.colors = o3d.utility.Vector3dVector(intensity_fake_rgb)
        o3d.io.write_point_cloud(save_filepath, pc)

    def validate_bin_pointcloud(self, bin_file: str) -> bool:
        try:
            data = np.fromfile(bin_file, dtype=np.float32)
            
            # Check if the file size is divisible by 5 (x,y,z,intensity,ring_index)
            if len(data) % 5 != 0:
                return False
                
            points = data.reshape(-1, 5)
            if len(points) == 0:
                return False
                
            if not (
                np.all(points[:, 3] >= 0) and  # intensity
                np.all(points[:, 3] <= 1) and  # intensity
                np.all(points[:, 4] >= 0) and  # ring_index
                np.all(points[:, 4] == points[:, 4].astype(int))  # ring_index should be integer
            ):
                return False
                
            return True
            
        except Exception:
            return False

    def _collect_items_if_format_not_detected(self) -> Tuple[List[Item], bool, Set[str]]:
        only_modality_items = True
        unsupported_exts = set()
        pcd_list, rimg_dict, rimg_ann_dict = [], {}, {}
        used_img_ext = set()
        
        bin_files = list_files_recursively(self._input_data, [".bin"], None, True)
        for bin_file in bin_files:
            if self.validate_bin_pointcloud(bin_file):
                self._convert_bin_to_pcd(bin_file, bin_file.replace(".bin", ".pcd"))
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                if file in ["key_id_map.json", "meta.json"]:
                    continue

                ext = get_file_ext(full_path)
                if ext == ".json":
                    dir_name = os.path.basename(root)
                    parent_dir_name = os.path.basename(os.path.dirname(root))
                    if any(
                        p.replace("_", " ") in ["images", "related images", "photo context"]
                        for p in [dir_name, parent_dir_name]
                    ) or dir_name.endswith("_pcd"):
                        rimg_ann_dict[file] = full_path
                elif imghdr.what(full_path):
                    rimg_dict[file] = full_path
                    if ext not in used_img_ext:
                        used_img_ext.add(ext)
                elif ext.lower() in self.allowed_exts:
                    try:
                        validate_pcd_ext(ext)
                        pcd_list.append(full_path)
                    except:
                        pass
                else:
                    only_modality_items = False
                    unsupported_exts.add(ext)

        # create Items
        items = []
        for pcd_path in pcd_list:
            item = self.Item(pcd_path)
            rimg, rimg_ann = helpers.find_related_items(
                item.name, used_img_ext, rimg_dict, rimg_ann_dict
            )
            if rimg is not None and rimg_ann is not None:
                item.set_related_images((rimg, rimg_ann))
            items.append(item)
        return items, only_modality_items, unsupported_exts
