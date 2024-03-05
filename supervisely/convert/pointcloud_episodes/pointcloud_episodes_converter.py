from typing import List, Optional

from tqdm import tqdm

from supervisely import Api, PointcloudEpisodeAnnotation, ProjectMeta, batched, generate_free_name, logger
from supervisely.api.module_api import ApiField
from supervisely.convert.base_converter import BaseConverter
from supervisely.io.json import load_json_file
from supervisely.pointcloud.pointcloud import ALLOWED_POINTCLOUD_EXTENSIONS


class PointcloudEpisodeConverter(BaseConverter):
    allowed_exts = ALLOWED_POINTCLOUD_EXTENSIONS

    class Item(BaseConverter.BaseItem):
        def __init__(
            self,
            item_path,
            frame_number: int,
            ann_data: Optional[str] = None,
            related_images: Optional[list] = None,
            custom_data: Optional[dict] = None,
        ):
            self._path = item_path
            self._frame_number = frame_number
            self._ann_data = ann_data
            self._related_images = related_images if related_images is not None else []
            self._type = "point_cloud_episode"
            self._custom_data = custom_data if custom_data is not None else {}

        @property
        def frame_count(self) -> int:
            return self._frame_count

        def create_empty_annotation(self) -> PointcloudEpisodeAnnotation:
            return PointcloudEpisodeAnnotation()

        def set_related_images(self, related_images: dict) -> None:
            self._related_images.append(related_images)

    def __init__(self, input_data: str):
        self._input_data: str = input_data
        self._items: List[self.Item] = []
        self._meta: ProjectMeta = None
        self._annotation = None
        self._frame_pointcloud_map = None
        self._frame_count = None
        self._converter = self._detect_format()

    @property
    def format(self):
        return self.converter.format

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

        dataset = api.dataset.get_info_by_id(dataset_id)
        existing_names = [pcde.name for pcde in api.image.get_list(dataset.id)]
        if self._meta is not None:
            curr_meta = self._meta
        else:
            curr_meta = ProjectMeta()
        meta_json = api.project.get_meta(dataset.project_id)
        meta = ProjectMeta.from_json(meta_json)
        meta = meta.merge(curr_meta)
        api.project.update_meta(dataset.project_id, meta)

        if log_progress:
            progress = tqdm(total=self.items_count, desc=f"Uploading pointcloud episodes...")
            progress_cb = progress.update
        else:
            progress_cb = None

        frame_to_pointcloud_ids = {}
        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            item_metas = []
            for item in batch:
                if item.name in existing_names:
                    new_name = generate_free_name(
                        existing_names, item.name, with_ext=True, extend_used_names=True
                    )
                    logger.warn(
                        f"Video with name '{item.name}' already exists, renaming to '{new_name}'"
                    )
                    item_names.append(new_name)
                else:
                    item_names.append(item.name)
                item_paths.append(item.path)

            pcd_infos = api.pointcloud_episode.upload_paths(
                dataset_id,
                item_names,
                item_paths,
                progress_cb=progress_cb,
            )
            pcd_ids = [pcd_info.id for pcd_info in pcd_infos]

            for item, pcd_id in zip(batch, pcd_ids):
                frame_to_pointcloud_ids[item._frame_number] = pcd_id
                item_metas.append({"frame": item._frame_number})
                rimg_infos = []
                camera_names = []
                img_paths, rimg_ann_paths = list(zip(*item._related_images))
                rimg_hashes = api.pointcloud_episode.upload_related_images(img_paths)
                for img_ind, (img_hash, rimg_ann_path) in enumerate(
                    zip(rimg_hashes, rimg_ann_paths)
                ):
                    meta_json = load_json_file(rimg_ann_path)
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
                        api.pointcloud.add_related_images(rimg_infos, camera_names)
                    except Exception as e:
                        logger.warn(
                            f"Failed to upload related image or add it to pointcloud episo: {repr(e)}"
                        )
                        continue

        if self.items_count > 0:
            ann = self.to_supervisely(self._items[0], meta)
            api.pointcloud_episode.annotation.append(dataset.id, ann, frame_to_pointcloud_ids)

        if log_progress:
            progress.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")
