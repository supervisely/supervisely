import mimetypes
import os
from pathlib import Path
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
from supervisely.api.api import ApiContext
from supervisely.convert.base_converter import BaseConverter
from supervisely.imaging.image import SUPPORTED_IMG_EXTS, is_valid_ext
from supervisely.io.fs import dirs_filter, get_file_ext, get_file_name, list_files
from supervisely.io.json import load_json_file
from supervisely.project.project_settings import LabelingInterface

# Images without annotations: map input folders to nested datasets (True) or put every image
# into the target dataset (False). Changing it changes what Auto Import does for its users.
DEFAULT_FOLDERS_AS_DATASETS = False

FOLDER_ITEMS = "items"
FOLDER_DATASETS = "datasets"


def _collapse_top_levels(chains: List[List[str]]) -> int:
    """Number of leading single-folder levels to drop from the top of the tree.

    Stops when an image lies directly at the current level, the level has more than one
    folder, or its single folder holds images itself (same rule as Import Images v1.3.0).
    """
    k = 0
    while True:
        cur = [c[k:] for c in chains]
        if not cur or any(len(c) == 0 for c in cur):
            return k
        tops = {c[0] for c in cur}
        if len(tops) != 1:
            return k
        top = next(iter(tops))
        if any(c == [top] for c in cur):
            return k
        k += 1


def group_items_by_folders(items: List[BaseConverter.BaseItem], input_root: str) -> Dict:
    """Group items into a dataset tree that follows the folders they lie in.

    The folder the input was copied to (one per input path) is the root, as the selected folder
    is in Import Images; when there is a single input, its folder is not a dataset. Leading
    folders that only hold one subfolder are dropped. Returns
    ``{"items": [root items], "datasets": {folder: {"items": [...], "datasets": {...}}}}``.
    """
    chains = []
    for item in items:
        rel = os.path.relpath(os.path.dirname(os.path.abspath(item.path)), input_root)
        chains.append([] if rel in (".", "") else list(Path(rel).parts))
    if chains and all(chains) and len({c[0] for c in chains}) == 1:
        chains = [c[1:] for c in chains]
    k = _collapse_top_levels(chains)
    tree = {FOLDER_ITEMS: [], FOLDER_DATASETS: {}}
    for item, chain in zip(items, chains):
        node = tree
        for part in chain[k:]:
            node = node[FOLDER_DATASETS].setdefault(part, {FOLDER_ITEMS: [], FOLDER_DATASETS: {}})
        node[FOLDER_ITEMS].append(item)
    return tree


class ImageConverter(BaseConverter):
    """Base converter for image projects (collects images and uploads images + annotations to Supervisely)."""

    allowed_exts = [
        ext for ext in SUPPORTED_IMG_EXTS + image_helper.EXT_TO_CONVERT if ext != ".nrrd"
    ]
    modality = "images"
    # set on the converter that collected plain images because no annotation format was found
    _plain_images_input = False
    _folders_as_datasets = None

    class Item(BaseConverter.BaseItem):
        """Base image item used by image converters (path + optional annotation/meta and image shape)."""

        def __init__(
            self,
            item_path: str,
            ann_data: Union[str, dict] = None,
            meta_data: Union[str, dict] = None,
            shape: Optional[Union[Tuple, List]] = None,
            custom_data: Optional[dict] = None,
        ):
            """
            :param item_path: Path to image file.
            :type item_path: str
            :param ann_data: Annotation (path or dict).
            :type ann_data: Union[str, dict], optional
            :param meta_data: Meta data (path or dict).
            :type meta_data: Union[str, dict], optional
            :param shape: Image shape (height, width).
            :type shape: Union[Tuple, List], optional
            :param custom_data: Extra per-item data.
            :type custom_data: dict, optional
            """
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

        def set_shape(self, shape: Optional[Tuple[int, int]] = None) -> None:
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
    def folders_as_datasets(self) -> bool:
        """Whether images without annotations are uploaded into datasets that follow their folders."""
        if self._folders_as_datasets is None:
            return DEFAULT_FOLDERS_AS_DATASETS
        return self._folders_as_datasets

    @folders_as_datasets.setter
    def folders_as_datasets(self, value: bool) -> None:
        self._folders_as_datasets = value

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
        progress_cb=None,
    ) -> None:
        """Upload converted data to Supervisely"""
        if entities is None and self._plain_images_input and self.folders_as_datasets:
            tree = group_items_by_folders(self._items, self._input_data)
            if tree[FOLDER_DATASETS]:
                return self._upload_folders_as_datasets(
                    api, dataset_id, tree, batch_size, log_progress, progress_cb
                )

        dataset_info = api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_id = dataset_info.project_id

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        existing_names = set([img.name for img in api.image.get_list(dataset_id)])
        progress = None
        if progress_cb is not None:
            log_progress = True
        elif log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading")

        if self.upload_as_links:
            batch_size = 1000

        for batch in batched(entities or self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            item_metas = []
            valid_batch_items = []
            for item in batch:
                item.path = self.validate_image(item.path)
                if item.path is None:
                    logger.warning(f"Image {item.path} failed validation and will be skipped.")
                    continue  # image has failed validation
                valid_batch_items.append(item)
                name = f"{get_file_name(item.path)}{get_file_ext(item.path).lower()}"

                item.name = generate_free_name(
                    existing_names, name, with_ext=True, extend_used_names=True
                )
                item_names.append(item.name)
                item_paths.append(item.path)

                if isinstance(item.meta, str):  # path to file
                    item_metas.append(load_json_file(item.meta))
                elif isinstance(item.meta, dict):
                    item_metas.append(item.meta)
                else:
                    item_metas.append({})

            if self._items and len(valid_batch_items) == 0:
                logger.warning("All items in the batch failed validation and will be skipped.")
                if log_progress:
                    progress_cb(len(batch))
                continue

            with ApiContext(
                api=api, project_id=project_id, dataset_id=dataset_id, project_meta=meta
            ):
                if self.upload_as_links:
                    img_infos = api.image.upload_links(
                        dataset_id,
                        item_names,
                        item_paths,
                        metas=item_metas,
                        batch_size=batch_size,
                        conflict_resolution="rename",
                        force_metadata_for_links=self._force_shape_for_links,
                    )
                else:
                    img_infos = api.image.upload_paths(
                        dataset_id,
                        item_names,
                        item_paths,
                        metas=item_metas,
                        conflict_resolution="rename",
                    )
                img_ids = [img_info.id for img_info in img_infos]

                if self.upload_as_links and not self.supports_links:
                    continue

                anns = []
                for info, item in zip(img_infos, valid_batch_items):
                    if self._force_shape_for_links:
                        item.set_shape((info.height, info.width))
                    anns.append(self.to_supervisely(item, meta, renamed_classes, renamed_tags))

                if len(anns) == len(img_ids):
                    api.annotation.upload_anns(
                        img_ids, anns, skip_bounds_validation=self.upload_as_links
                    )

            if log_progress:
                progress_cb(len(batch))

        if log_progress:
            if is_development() and progress is not None:
                progress.close()
        logger.info(
            f"Dataset has been successfully uploaded → {dataset_info.name}, ID:{dataset_id}"
        )

    def _upload_folders_as_datasets(
        self,
        api: Api,
        dataset_id: int,
        tree: Dict,
        batch_size: int,
        log_progress: bool,
        progress_cb=None,
    ) -> None:
        """Upload a folder tree from :func:`group_items_by_folders` as nested datasets.

        Images at the input root go into the target dataset and folders become datasets nested
        inside it, as for Import Images into an existing dataset. The target dataset is never
        renamed and nothing is created outside it: the converter cannot tell a dataset the caller
        just created from an empty dataset the user picked.
        """
        target = api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_id = target.project_id
        progress = None
        if progress_cb is None and log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading")

        def _upload_node(node: Dict, ds_id: int, ds_path: str) -> None:
            items = node[FOLDER_ITEMS]
            logger.info(
                f"Dataset: {ds_path}, items: {len(items)}, "
                f"nested datasets: {len(node[FOLDER_DATASETS])}"
            )
            if items:  # an empty list would make upload_dataset take all items
                ImageConverter.upload_dataset(
                    self,
                    api,
                    ds_id,
                    batch_size,
                    log_progress=log_progress,
                    entities=items,
                    progress_cb=progress_cb,
                )
            for name in sorted(node[FOLDER_DATASETS]):
                child = api.dataset.create(
                    project_id, name, parent_id=ds_id, change_name_if_conflict=True
                )
                _upload_node(node[FOLDER_DATASETS][name], child.id, f"{ds_path}/{child.name}")

        logger.info("Uploading images into datasets that follow the input folders")
        _upload_node(tree, target.id, target.name)

        if is_development() and progress is not None:
            progress.close()

    def validate_image(self, path: str) -> str:
        if self.upload_as_links:
            return self.remote_files_map.get(path)
        return image_helper.validate_image(path)

    def is_image(self, path: str) -> bool:
        if self._upload_as_links and self.supports_links:
            ext = get_file_ext(path)
            return ext.lower() in self.allowed_exts
        mimetypes.add_type("image/heic", ".heic")  # to extend types_map
        mimetypes.add_type("image/heif", ".heif")  # to extend types_map
        mimetypes.add_type("image/jpeg", ".jfif")  # to extend types_map
        mimetypes.add_type("image/avif", ".avif")  # to extend types_map
        mimetypes.add_type("image/bmp", ".bmp")  # to extend types_map

        # libmagic reads only the file header; resolve symlinks, which it does not follow
        mimetype = magic.from_file(os.path.realpath(path), mime=True)
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
                jsons = list_files(
                    dirpath,
                    valid_extensions=[".json"],
                    ignore_valid_extensions_case=True,
                )
                return len(jsons) > 0
            return False

        meta_dirs = [d for d in dirs_filter(self._input_data, _is_meta_dir)]
        if len(meta_dirs) == 0:
            self._plain_images_input = True
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
