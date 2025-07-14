import imghdr
import os
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

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
from supervisely.io.fs import get_file_ext, get_file_name_with_ext
from supervisely.io.json import load_json_file
from supervisely.pointcloud.pointcloud import ALLOWED_POINTCLOUD_EXTENSIONS
from supervisely.pointcloud.pointcloud import validate_ext as validate_pcd_ext
from supervisely.pointcloud_annotation.constants import OBJECT_KEY
from supervisely.video_annotation.key_id_map import KeyIdMap


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

        def set_related_images(self, related_images: Tuple[str, str, Optional[str]]) -> None:
            """Adds related image to the item.

            related_images tuple:
                - path to image
                - path to .json with image metadata
                - path to .figures.json (can be None if no figures)
            """
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

        key_id_map = KeyIdMap()
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
            pcl_to_rimg_figures: Dict[int, Dict[str, List[Dict]]] = {}
            pcl_to_hash_to_id: Dict[int, Dict[str, int]] = {}
            for pcd_id, ann, item in zip(pcd_ids, anns, batch):
                if ann is not None:
                    api.pointcloud.annotation.append(pcd_id, ann, key_id_map)

                rimg_infos = []
                camera_names = []
                for img_ind, rel_tuple in enumerate(item._related_images):
                    img_path = rel_tuple[0]
                    rimg_ann_path = rel_tuple[1]
                    fig_path = rel_tuple[2] if len(rel_tuple) > 2 else None
                    meta_json = load_json_file(rimg_ann_path)
                    try:
                        if ApiField.META not in meta_json:
                            raise ValueError("Related image meta not found in json file.")
                        if ApiField.NAME not in meta_json:
                            raise ValueError("Related image name not found in json file.")
                        img_hash = api.pointcloud.upload_related_image(img_path)
                        if "deviceId" not in meta_json[ApiField.META].keys():
                            camera_names.append(f"CAM_{str(img_ind).zfill(2)}")
                        else:
                            camera_names.append(meta_json[ApiField.META]["deviceId"])
                        rimg_infos.append(
                            {
                                ApiField.ENTITY_ID: pcd_id,
                                ApiField.NAME: meta_json[ApiField.NAME],
                                ApiField.HASH: img_hash,
                                ApiField.META: meta_json[ApiField.META],
                            }
                        )

                        if fig_path is not None and os.path.isfile(fig_path):
                            try:
                                figs_json = load_json_file(fig_path)
                                pcl_to_rimg_figures.setdefault(pcd_id, {})[img_hash] = figs_json
                            except Exception as e:
                                logger.debug(f"Failed to read figures json '{fig_path}': {repr(e)}")

                    except Exception as e:
                        logger.warn(
                            f"Failed to upload related image or add it to pointcloud: {repr(e)}"
                        )
                        continue

                # add images for this point cloud
                if len(rimg_infos) > 0:
                    try:
                        uploaded_rimgs = api.pointcloud.add_related_images(rimg_infos, camera_names)
                        # build mapping hash->id
                        for info, uploaded in zip(rimg_infos, uploaded_rimgs):
                            img_hash = info.get(ApiField.HASH)
                            img_id = (
                                uploaded.get(ApiField.ID)
                                if isinstance(uploaded, dict)
                                else getattr(uploaded, "id", None)
                            )
                            if img_hash is not None and img_id is not None:
                                pcl_to_hash_to_id.setdefault(pcd_id, {})[img_hash] = img_id
                    except Exception as e:
                        logger.debug(f"Failed to add related images to pointcloud: {repr(e)}")

            # ---- upload figures for processed batch ----
            if len(pcl_to_rimg_figures) > 0:
                try:
                    dataset_info = api.dataset.get_info_by_id(dataset_id)
                    project_id = dataset_info.project_id

                    figures_payload: List[Dict] = []

                    for pcl_id, hash_to_figs in pcl_to_rimg_figures.items():
                        hash_to_ids = pcl_to_hash_to_id.get(pcl_id, {})
                        if len(hash_to_ids) == 0:
                            continue

                        for img_hash, figs_json in hash_to_figs.items():
                            if img_hash not in hash_to_ids:
                                continue
                            rimg_id = hash_to_ids[img_hash]
                            for fig in figs_json:
                                try:
                                    fig[ApiField.ENTITY_ID] = rimg_id
                                    fig[ApiField.DATASET_ID] = dataset_id
                                    fig[ApiField.PROJECT_ID] = project_id
                                    if OBJECT_KEY in fig:
                                        fig[ApiField.OBJECT_ID] = key_id_map.get_object_id(
                                            UUID(fig[OBJECT_KEY])
                                        )
                                except Exception as e:
                                    logger.debug(
                                        f"Failed to process figure json for img_hash={img_hash}: {repr(e)}"
                                    )
                                    continue

                            figures_payload.extend(figs_json)

                        if len(figures_payload) > 0:
                            try:
                                api.image.figure.create_bulk(
                                    figures_json=figures_payload, dataset_id=dataset_id
                                )
                            except Exception as e:
                                logger.debug(
                                    f"Failed to upload figures for related images: {repr(e)}"
                                )
                except Exception as e:
                    logger.debug(f"Unexpected error during related image figures upload: {repr(e)}")

            if log_progress:
                progress_cb(len(batch))

        if log_progress:
            if is_development():
                progress.close()
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")

    def _collect_items_if_format_not_detected(self) -> Tuple[List[Item], bool, Set[str]]:
        only_modality_items = True
        unsupported_exts = set()
        pcd_list, rimg_dict, rimg_ann_dict, rimg_fig_dict = [], {}, {}, {}
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                if file in ["key_id_map.json", "meta.json"]:
                    continue

                ext = get_file_ext(full_path)
                if file.endswith(".figures.json"):
                    rimg_fig_dict[file] = full_path
                elif ext == ".json":
                    dir_name = os.path.basename(root)
                    parent_dir_name = os.path.basename(os.path.dirname(root))
                    if any(
                        p.replace("_", " ") in ["images", "related images", "photo context"]
                        for p in [dir_name, parent_dir_name]
                    ) or dir_name.endswith("_pcd"):
                        rimg_ann_dict[file] = full_path
                elif imghdr.what(full_path):
                    dir_name = os.path.basename(root)
                    if dir_name not in rimg_dict:
                        rimg_dict[dir_name] = []
                    rimg_dict[dir_name].append(full_path)
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
            rimg_dir_name = item.name.replace(".pcd", "_pcd")
            rimgs = rimg_dict.get(rimg_dir_name, [])
            for rimg_path in rimgs:
                rimg_ann_name = f"{get_file_name_with_ext(rimg_path)}.json"
                if rimg_ann_name in rimg_ann_dict:
                    rimg_ann_path = rimg_ann_dict[rimg_ann_name]
                    rimg_fig_name = f"{get_file_name_with_ext(rimg_path)}.figures.json"
                    rimg_fig_path = rimg_fig_dict.get(rimg_fig_name, None)
                    if rimg_fig_path is not None and not os.path.exists(rimg_fig_path):
                        rimg_fig_path = None
                    item.set_related_images((rimg_path, rimg_ann_path, rimg_fig_path))
            items.append(item)
        return items, only_modality_items, unsupported_exts
