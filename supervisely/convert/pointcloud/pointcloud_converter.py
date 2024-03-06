from typing import List, Optional
from tqdm import tqdm

from supervisely import Api, batched, generate_free_name, logger, PointcloudAnnotation, ProjectMeta
from supervisely.api.module_api import ApiField
from supervisely.convert.base_converter import BaseConverter
from supervisely.io.json import load_json_file
from supervisely.pointcloud.pointcloud import ALLOWED_POINTCLOUD_EXTENSIONS


class PointcloudConverter(BaseConverter):
    allowed_exts = ALLOWED_POINTCLOUD_EXTENSIONS

    class Item(BaseConverter.BaseItem):
        def __init__(
            self,
            item_path,
            ann_data=None,
            related_images: Optional[list] = None,
            custom_data: Optional[dict] = None,
        ):
            self._path = item_path
            self._ann_data = ann_data
            self._type = "point_cloud"
            self._related_images = related_images if related_images is not None else []
            self._custom_data = custom_data if custom_data is not None else {}

        @property
        def frame_count(self) -> int:
            return self._frame_count

        def create_empty_annotation(self) -> PointcloudAnnotation:
            return PointcloudAnnotation()

        def set_related_images(self, related_images: dict) -> None:
            self._related_images.append(related_images)

    def __init__(self, input_data: str):
        self._input_data: str = input_data
        self._meta: ProjectMeta = None
        self._items: List[self.Item] = []
        self._converter = self._detect_format()
        self._batch_size = 1

    @property
    def format(self):
        return self.converter.format

    @property
    def ann_ext(self):
        return None

    @property
    def key_file_ext(self):
        return None

    def get_meta(self) -> ProjectMeta:
        return self._meta

    def get_items(self) -> List[BaseConverter.BaseItem]:
        return self._items

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
        existing_names = set([pcd.name for pcd in api.pointcloud.get_list(dataset.id)])
        if self._meta is not None:
            curr_meta = self._meta
        else:
            curr_meta = ProjectMeta()
        meta_json = api.project.get_meta(dataset.project_id)
        meta = ProjectMeta.from_json(meta_json)
        meta = meta.merge(curr_meta)
        api.project.update_meta(dataset.project_id, meta)

        if log_progress:
            progress = tqdm(total=self.items_count, desc=f"Uploading pointclouds...")
            progress_cb = progress.update
        else:
            progress_cb = None

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            anns = []
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

                ann = self.to_supervisely(item, meta)
                anns.append(ann)

            pcd_infos = api.pointcloud.upload_paths(
                dataset_id,
                item_names,
                item_paths,
                progress_cb=progress_cb,
            )
            pcd_ids = [pcd_info.id for pcd_info in pcd_infos]

            for pcd_id, ann in zip(pcd_ids, anns):
                api.pointcloud.annotation.append(pcd_id, ann)

                rimg_infos = []
                camera_names = []
                for img_ind, (img_path, rimg_ann_path) in enumerate(item._related_images):
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
            progress.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")
