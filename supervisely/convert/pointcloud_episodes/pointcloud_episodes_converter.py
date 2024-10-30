from typing import Dict, Optional, Union

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
from supervisely.io.json import load_json_file
from supervisely.project.project_settings import LabelingInterface
from supervisely.pointcloud.pointcloud import ALLOWED_POINTCLOUD_EXTENSIONS


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

        def set_related_images(self, related_images: dict) -> None:
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

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(
            api, dataset_id
        )

        existing_names = set(
            [pcde.name for pcde in api.pointcloud_episode.get_list(dataset_id)]
        )

        if log_progress:
            progress, progress_cb = self.get_progress(
                self.items_count, "Uploading pointcloud episodes..."
            )
        else:
            progress_cb = None

        frame_to_pointcloud_ids = {}
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
                    img_paths, rimg_ann_paths = list(zip(*item._related_images))
                    rimg_hashes = api.pointcloud_episode.upload_related_images(
                        img_paths
                    )
                    for img_ind, (img_hash, rimg_ann_path) in enumerate(
                        zip(rimg_hashes, rimg_ann_paths)
                    ):
                        meta_json = load_json_file(rimg_ann_path)
                        try:
                            if ApiField.META not in meta_json:
                                raise ValueError(
                                    "Related image meta not found in json file."
                                )
                            if ApiField.NAME not in meta_json:
                                raise ValueError(
                                    "Related image name not found in json file."
                                )
                            if "deviceId" not in meta_json[ApiField.META].keys():
                                camera_names.append(f"CAM_{str(img_ind).zfill(2)}")
                            else:
                                camera_names.append(
                                    meta_json[ApiField.META]["deviceId"]
                                )
                            rimg_infos.append(
                                {
                                    ApiField.ENTITY_ID: pcd_id,
                                    ApiField.NAME: meta_json[ApiField.NAME],
                                    ApiField.HASH: img_hash,
                                    ApiField.META: meta_json[ApiField.META],
                                }
                            )
                            api.pointcloud.add_related_images(rimg_infos, camera_names)
                        except Exception as e:
                            logger.warn(
                                f"Failed to upload related image or add it to pointcloud episo: {repr(e)}"
                            )
                            continue

            if log_progress:
                progress_cb(len(batch))

        if self.items_count > 0:
            ann = self.to_supervisely(
                self._items[0], meta, renamed_classes, renamed_tags
            )
            api.pointcloud_episode.annotation.append(
                dataset_id, ann, frame_to_pointcloud_ids
            )

        if log_progress:
            if is_development():
                progress.close()
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")
