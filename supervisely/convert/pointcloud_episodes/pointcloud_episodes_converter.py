import imghdr
import os
from typing import Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

from supervisely import (
    Api,
    PointcloudEpisodeAnnotation,
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
from supervisely.project.project_settings import LabelingInterface
from supervisely.video_annotation.key_id_map import KeyIdMap


class PointcloudEpisodeConverter(BaseConverter):
    allowed_exts = ALLOWED_POINTCLOUD_EXTENSIONS
    modality = "pointcloud episodes"

    class Item(BaseConverter.BaseItem):
        def __init__(
            self,
            item_path,
            frame_number: int,
            ann_data: Optional[str] = None,
            related_images: Optional[list] = None,
            custom_data: Optional[dict] = None,
        ):
            self._name: str = None
            self._path = item_path
            self._frame_number = frame_number
            self._ann_data = ann_data
            self._related_images = related_images if related_images is not None else []
            self._type = "point_cloud_episode"
            self._custom_data = custom_data if custom_data is not None else {}

        @property
        def frame_number(self) -> int:
            return self._frame_number

        def create_empty_annotation(self) -> PointcloudEpisodeAnnotation:
            return PointcloudEpisodeAnnotation()

        def set_related_images(self, related_images: Tuple[str, str, Optional[str]]) -> None:
            """Adds related image to the item.

            related_images tuple:
                - path to image
                - path to .json with image metadata
                - path to .figures.json (can be None if no figures)
            """
            self._related_images.append(related_images)

    def __init__(
        self,
        input_data: str,
        labeling_interface: Optional[Union[LabelingInterface, str]],
        upload_as_links: bool = False,
        remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)
        self._annotation = None
        self._frame_pointcloud_map = None
        self._frame_count = None

    @property
    def format(self):
        return self._converter.format

    @property
    def frame_count(self):
        if self._frame_count is None:
            self._frame_count = len(self._items)
        return self._frame_count

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

        existing_names = set([pcde.name for pcde in api.pointcloud_episode.get_list(dataset_id)])

        if log_progress:
            progress, progress_cb = self.get_progress(
                self.items_count, "Uploading pointcloud episodes..."
            )
        else:
            progress_cb = None

        frame_to_pointcloud_ids: Dict[int, int] = {}
        pcl_to_rimg_figures: Dict[int, Dict[str, List[Dict]]] = {}
        pcl_to_hash_to_id: Dict[int, Dict[str, int]] = {}
        key_id_map = KeyIdMap()
        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            item_metas = []
            for item in batch:
                item.name = generate_free_name(
                    existing_names, item.name, with_ext=True, extend_used_names=True
                )
                item_names.append(item.name)
                item_paths.append(item.path)

            pcd_infos = api.pointcloud_episode.upload_paths(
                dataset_id,
                item_names,
                item_paths,
            )
            pcd_ids = [pcd_info.id for pcd_info in pcd_infos]

            for item, pcd_id in zip(batch, pcd_ids):
                frame_to_pointcloud_ids[item._frame_number] = pcd_id
                item_metas.append({"frame": item._frame_number})
                rimg_infos = []
                camera_names = []
                if len(item._related_images) > 0:
                    img_paths: List[str] = []
                    ann_paths: List[str] = []
                    fig_paths: List[Optional[str]] = []

                    for triple in item._related_images:
                        img_paths.append(triple[0])
                        ann_paths.append(triple[1])
                        fig_paths.append(triple[2] if len(triple) > 2 else None)

                    rimg_hashes = api.pointcloud_episode.upload_related_images(img_paths)

                    for img_ind, (img_hash, ann_path, fig_path) in enumerate(
                        zip(rimg_hashes, ann_paths, fig_paths)
                    ):
                        meta_json = load_json_file(ann_path)
                        try:
                            if ApiField.META not in meta_json:
                                raise ValueError("Related image meta not found in json file.")
                            if ApiField.NAME not in meta_json:
                                raise ValueError("Related image name not found in json file.")
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

                            if fig_path is not None:
                                try:
                                    figs_json = load_json_file(fig_path)
                                    pcl_to_rimg_figures.setdefault(pcd_id, {})[img_hash] = figs_json
                                except Exception as e:
                                    logger.debug(
                                        f"Failed to read figures json '{fig_path}': {repr(e)}"
                                    )

                        except Exception as e:
                            logger.warning(
                                f"Failed to process related image meta '{ann_path}': {repr(e)}"
                            )
                            continue

                    uploaded_rimgs = api.pointcloud.add_related_images(rimg_infos, camera_names)
                    try:
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
                        logger.debug(
                            f"Failed to build hash->ID mapping for related images: {repr(e)}"
                        )

            if log_progress:
                progress_cb(len(batch))

        if self.items_count > 0:
            ann = self.to_supervisely(self._items[0], meta, renamed_classes, renamed_tags)
            api.pointcloud_episode.annotation.append(
                dataset_id, ann, frame_to_pointcloud_ids, key_id_map
            )

        # ---- upload figures for processed batch ----
        if len(pcl_to_rimg_figures) > 0:
            if log_progress:
                progress, progress_cb = self.get_progress(
                    self.items_count, "Uploading figures for related images..."
                )
            else:
                progress_cb = None

            try:
                dataset_info = api.dataset.get_info_by_id(dataset_id)
                project_id = dataset_info.project_id

                figures_to_upload: List[Dict] = []
                for pcl_id, hash_to_figs in pcl_to_rimg_figures.items():
                    hash_to_ids = pcl_to_hash_to_id.get(pcl_id, {})
                    if len(hash_to_ids) == 0:
                        continue

                    for img_hash, figs_json in hash_to_figs.items():
                        if img_hash not in hash_to_ids:
                            continue
                        img_id = hash_to_ids[img_hash]

                        for fig in figs_json:
                            try:
                                fig[ApiField.ENTITY_ID] = img_id
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

                        figures_to_upload.extend(figs_json)

                if len(figures_to_upload) > 0:
                    try:
                        api.image.figure.create_bulk(
                            figures_json=figures_to_upload, dataset_id=dataset_id
                        )
                    except Exception as e:
                        logger.debug(f"Failed to upload figures for related images: {repr(e)}")

                if log_progress:
                    progress_cb(len(pcl_to_rimg_figures))

            except Exception as e:
                logger.debug(f"Unexpected error during figures upload: {repr(e)}")

        if log_progress:
            if is_development():
                progress.close()
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")

    def _collect_items_if_format_not_detected(self) -> Tuple[List[Item], bool, Set[str]]:
        only_modality_items = True
        unsupported_exts = set()
        pcd_dict = {}
        frames_pcd_map = None
        rimg_dict, rimg_json_dict, rimg_fig_dict = {}, {}, {}
        for root, _, files in os.walk(self._input_data):
            dir_name = os.path.basename(root)
            for file in files:
                full_path = os.path.join(root, file)
                if file == "frame_pointcloud_map.json":
                    frames_pcd_map = load_json_file(full_path)
                    continue

                ext = get_file_ext(full_path)
                recognized_ext = imghdr.what(full_path)
                if file.endswith(".figures.json"):
                    rimg_fig_dict[file] = full_path
                elif ext == ".json":
                    rimg_json_dict[file] = full_path
                elif recognized_ext:
                    if ext.lower() == ".pcd":
                        logger.warning(
                            f"File '{file}' has been recognized as '.{recognized_ext}' format. Skipping."
                        )
                        continue
                    if dir_name not in rimg_dict:
                        rimg_dict[dir_name] = []
                    rimg_dict[dir_name].append(full_path)
                elif ext.lower() in self.allowed_exts:
                    try:
                        validate_pcd_ext(ext)
                        pcd_dict[file] = full_path
                    except:
                        pass
                else:
                    only_modality_items = False
                    unsupported_exts.add(ext)

        items = []
        updated_frames_pcd_map = {}
        if frames_pcd_map:
            list_of_pcd_names = list(frames_pcd_map.values())
        else:
            list_of_pcd_names = sorted(pcd_dict.keys())

        for i, pcd_name in enumerate(list_of_pcd_names):
            if pcd_name in pcd_dict:
                updated_frames_pcd_map[i] = pcd_name
                item = self.Item(pcd_dict[pcd_name], i)
                rimg_dir_name = pcd_name.replace(".pcd", "_pcd")
                rimgs = rimg_dict.get(rimg_dir_name, [])
                for rimg_path in rimgs:
                    rimg_ann_name = f"{get_file_name_with_ext(rimg_path)}.json"
                    if rimg_ann_name in rimg_json_dict:
                        rimg_ann_path = rimg_json_dict[rimg_ann_name]
                        rimg_fig_name = f"{get_file_name_with_ext(rimg_path)}.figures.json"
                        rimg_fig_path = rimg_fig_dict.get(rimg_fig_name, None)
                        if rimg_fig_path is not None and not os.path.exists(rimg_fig_path):
                            rimg_fig_path = None
                        item.set_related_images((rimg_path, rimg_ann_path, rimg_fig_path))
                items.append(item)
            else:
                logger.warning(f"Pointcloud file {pcd_name} not found. Skipping frame.")
                continue
        self._frame_pointcloud_map = updated_frames_pcd_map
        self._frame_count = len(self._frame_pointcloud_map)

        return items, only_modality_items, unsupported_exts
