import os
import mimetypes
from typing import Dict, List, Optional, Tuple, Union

import cv2
import magic
import nrrd

import supervisely.convert.image.image_helper as image_helper
from supervisely import (
    Annotation,
    Api,
    ProjectMeta,
    batched,
    generate_free_name,
    is_development,
    logger,
)
from supervisely.convert.base_converter import BaseConverter
from supervisely.api.api import ApiContext
from supervisely.imaging.image import SUPPORTED_IMG_EXTS, is_valid_ext
from supervisely.io.fs import get_file_ext, get_file_name, dirs_filter, list_files
from supervisely.io.json import load_json_file
from supervisely.project.project_settings import LabelingInterface


class ImageConverter(BaseConverter):
    allowed_exts = [
        ext for ext in SUPPORTED_IMG_EXTS + image_helper.EXT_TO_CONVERT if ext != ".nrrd"
    ]
    modality = "images"

    class Item(BaseConverter.BaseItem):
        def __init__(
            self,
            item_path: str,
            ann_data: Union[str, dict] = None,
            meta_data: Union[str, dict] = None,
            shape: Union[Tuple, List] = None,
            custom_data: Optional[dict] = None,
        ):
            self._path: str = item_path
            self._name: str = None
            self._ann_data: Union[str,] = ann_data
            self._meta_data: Union[str, dict] = meta_data
            self._type: str = "image"
            self._shape: Optional[Union[Tuple, List]] = shape
            self._custom_data: dict = custom_data if custom_data is not None else {}

        @property
        def meta(self) -> Union[str, dict]:
            return self._meta_data

        def set_shape(self, shape: Tuple[int, int] = None) -> None:
            try:
                if shape is not None:
                    self._shape = shape
                elif self._shape is None:
                    image = None
                    file_ext = get_file_ext(self.path).lower()
                    if file_ext == ".nrrd":
                        logger.debug(f"Found nrrd file: {self.path}.")
                        image, _ = nrrd.read(self.path)
                    elif file_ext in [".tif", ".tiff"]:
                        image = image_helper.read_tiff_image(self.path)
                    elif is_valid_ext(file_ext):
                        logger.debug(f"Found image file: {self.path}.")
                        image = cv2.imread(self.path)

                    if image is not None:
                        self._shape = image.shape[:2]
                    else:
                        self._shape = [0, 0]
            except Exception as e:
                logger.warning(f"Failed to read image shape: {e}, shape is set to [0, 0]")
                self._shape = [0, 0]

        def set_meta_data(self, meta_data: Union[str, dict]) -> None:
            self._meta_data = meta_data

        def create_empty_annotation(self) -> Annotation:
            if self._shape is None:
                self.set_shape()
            return Annotation(self._shape)

    @property
    def format(self) -> str:
        return self._converter.format

    @property
    def ann_ext(self) -> str:
        return None

    @property
    def key_file_ext(self) -> str:
        return None

    def validate_labeling_interface(self) -> bool:
        return self._labeling_interface in [
            LabelingInterface.DEFAULT,
            LabelingInterface.IMAGE_MATTING,
            LabelingInterface.FISHEYE,
        ]

    @staticmethod
    def validate_ann_file(ann_path: str, meta: ProjectMeta = None) -> bool:
        return False

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
        entities: List[Item] = None,
    ) -> None:
        """Upload converted data to Supervisely"""
        dataset_info = api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_id = dataset_info.project_id

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        existing_names = set([img.name for img in api.image.get_list(dataset_id)])
        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading images...")
        else:
            progress_cb = None

        for batch in batched(entities or self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            item_metas = []
            anns = []
            for item in batch:
                item.path = self.validate_image(item.path)
                if item.path is None:
                    continue  # image has failed validation
                item.name = f"{get_file_name(item.path)}{get_file_ext(item.path).lower()}"
                if self.upload_as_links:
                    ann = None  # TODO: implement
                else:
                    ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
                name = generate_free_name(
                    existing_names, item.name, with_ext=True, extend_used_names=True
                )
                item_names.append(name)
                item_paths.append(item.path)
                item_metas.append(load_json_file(item.meta) if item.meta else {})
                if ann is not None:
                    anns.append(ann)

            with ApiContext(
                api=api, project_id=project_id, dataset_id=dataset_id, project_meta=meta
            ):
                upload_method = (
                    api.image.upload_links if self.upload_as_links else api.image.upload_paths
                )
                img_infos = upload_method(
                    dataset_id,
                    item_names,
                    item_paths,
                    metas=item_metas,
                    conflict_resolution="rename",
                )
                img_ids = [img_info.id for img_info in img_infos]
                if len(anns) == len(img_ids):
                    api.annotation.upload_anns(img_ids, anns)

            if log_progress:
                progress_cb(len(batch))

        if log_progress:
            if is_development():
                progress.close()
        logger.info(f"Dataset ID:'{dataset_id}' has been successfully uploaded.")

    def validate_image(self, path: str) -> Tuple[str, str]:
        if self.upload_as_links:
            return self.remote_files_map.get(path)
        return image_helper.validate_image(path)

    def is_image(self, path: str) -> bool:
        mimetypes.add_type("image/heic", ".heic")  # to extend types_map
        mimetypes.add_type("image/heif", ".heif")  # to extend types_map
        mimetypes.add_type("image/jpeg", ".jfif")  # to extend types_map
        mimetypes.add_type("image/avif", ".avif")  # to extend types_map

        mime = magic.Magic(mime=True)
        mimetype = mime.from_file(path)
        file_ext = mimetypes.guess_extension(mimetype)
        if file_ext is None:
            return False
        else:
            if file_ext.lower() == ".bin" and get_file_ext(path).lower() == ".avif":
                return True
            return file_ext.lower() in self.allowed_exts

    def _collect_items_if_format_not_detected(self):

        def _is_meta_dir(dirpath: str) -> bool:
            if os.path.basename(dirpath).lower() == "meta":
                jsons = list_files(dirpath, valid_extensions=[".json"], ignore_valid_extensions_case=True)
                return len(jsons) > 0
            return False

        meta_dirs = [d for d in dirs_filter(self._input_data, _is_meta_dir)]
        if len(meta_dirs) == 0:
            return super()._collect_items_if_format_not_detected()
        else:
            logger.debug("Found folders with meta information for images.")
            only_modality_items = True
            unsupported_exts = set()
            images_map = {}
            metas_map = {}
            for root, _, files in os.walk(self._input_data):
                for file in files:
                    dirname = os.path.basename(root)
                    full_path = os.path.join(root, file)
                    ext = get_file_ext(full_path)
                    if ext.lower() in self.allowed_exts:  # pylint: disable=no-member
                        images_map[file] = full_path
                        continue
                    elif ext.lower() == ".json" and dirname == "meta":
                        file_name_no_ext = get_file_name(file)
                        metas_map[file_name_no_ext] = full_path
                        continue
                    only_modality_items = False
                    if ext.lower() in self.unsupported_exts:
                        unsupported_exts.add(ext)

            items = []
            for image_name, image_path in images_map.items():
                item = self.Item(image_path)
                meta_path = metas_map.get(image_name)
                if meta_path is not None:
                    item.set_meta_data(meta_path)
                items.append(item)

            return items, only_modality_items, unsupported_exts
