from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from supervisely._utils import batched, get_or_create_event_loop, is_production
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.tag_meta import TagValueType
from supervisely.api.api import Api
from supervisely.io.env import team_id
from supervisely.io.fs import (
    get_file_ext,
    get_file_name_with_ext,
    is_archive,
    remove_dir,
    silent_remove,
    unpack_archive,
)
from supervisely.annotation.obj_class import ObjClass
from supervisely.geometry.graph import GraphNodes
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_settings import LabelingInterface
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress


class AvailableImageConverters:
    SLY = "supervisely"
    COCO = "coco"
    FAST_COCO = "coco (fast)"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"
    CSV = "csv"
    MULTISPECTRAL = "multispectral"
    MASKS = "images_with_masks"
    MEDICAL2D = "medical_2d"
    MULTI_VIEW = "multi_view"
    PDF = "pdf"
    CITYSCAPES = "cityscapes"
    LABEL_ME = "label_me"
    LABEL_STUDIO = "label_studio"
    HIGH_COLOR_DEPTH = "high_color_depth"


class AvailableVideoConverters:
    SLY = "supervisely"
    MOT = "mot"
    DAVIS = "davis"


class AvailablePointcloudConverters:
    SLY = "supervisely"
    LAS = "las/laz"
    PLY = "ply"
    BAG = "rosbag"
    LYFT = "lyft"
    NUSCENES = "nuscenes"
    KITTI3D = "kitti3d"


class AvailablePointcloudEpisodesConverters:
    SLY = "supervisely"
    BAG = "rosbag"
    LYFT = "lyft"
    KITTI360 = "kitti360"


class AvailableVolumeConverters:
    SLY = "supervisely"
    DICOM = "dicom"
    NII = "nii"


class BaseConverter:
    unsupported_exts = [".gif", ".html", ".htm"]

    class BaseItem:
        def __init__(
            self,
            item_path: str,
            ann_data: Union[str, dict] = None,
            shape: Union[Tuple, List] = None,
            custom_data: Optional[dict] = None,
        ):
            self._path: str = item_path
            self._name: str = None
            self._ann_data: Union[str, dict, list] = ann_data
            self._shape: Union[Tuple, List] = shape
            self._custom_data: dict = custom_data or {}

        @property
        def name(self) -> str:
            if self._name is not None:
                return self._name
            return get_file_name_with_ext(self._path)

        @name.setter
        def name(self, name: str) -> None:
            self._name = name

        @property
        def path(self) -> str:
            return self._path

        @path.setter
        def path(self, path: str) -> None:
            self._path = path

        @property
        def ann_data(self) -> Union[str, dict]:
            return self._ann_data

        @ann_data.setter
        def ann_data(self, ann_data: Union[str, dict, list]) -> None:
            self._ann_data = ann_data

        @property
        def shape(self) -> Union[Tuple, List]:
            return self._shape

        @property
        def custom_data(self) -> dict:
            return self._custom_data

        @custom_data.setter
        def custom_data(self, custom_data: dict) -> None:
            self._custom_data = custom_data

        def set_shape(self, shape) -> None:
            self._shape = shape

        def update_custom_data(self, custom_data: dict) -> None:
            self.custom_data.update(custom_data)

        def update(
            self,
            item_path: str = None,
            ann_data: Union[str, dict] = None,
            shape: Union[Tuple, List] = None,
            custom_data: dict = {},
        ) -> None:
            if item_path is not None:
                self.path = item_path
            if ann_data is not None:
                self.ann_data = ann_data
            if shape is not None:
                self.set_shape(shape)
            if custom_data:
                self.update_custom_data(custom_data)

        def create_empty_annotation(self) -> Annotation:
            raise NotImplementedError()

    def __init__(
        self,
        input_data: str,
        labeling_interface: Optional[Union[LabelingInterface, str]] = LabelingInterface.DEFAULT,
        upload_as_links: bool = False,
        remote_files_map: Optional[Dict[str, str]] = None,
    ):
        self._input_data: str = input_data
        self._items: List[BaseConverter.BaseItem] = []
        self._meta: ProjectMeta = None
        self._labeling_interface = labeling_interface or LabelingInterface.DEFAULT

        # import as links settings
        self._upload_as_links: bool = upload_as_links
        self._remote_files_map: Optional[Dict[str, str]] = remote_files_map
        self._supports_links = False  # if converter supports uploading by links
        self._force_shape_for_links = False
        self._api = Api.from_env() if self._upload_as_links else None
        self._team_id = team_id() if self._upload_as_links else None
        self._converter = None

        if self._labeling_interface not in LabelingInterface.values():
            raise ValueError(
                f"Invalid labeling interface value: {labeling_interface}. "
                f"The available values: {LabelingInterface.values()}"
            )

    @property
    def format(self) -> str:
        return self.__str__()

    @property
    def items_count(self) -> int:
        return len(self._items)

    @property
    def ann_ext(self) -> str:
        raise NotImplementedError()

    @property
    def key_file_ext(self) -> str:
        raise NotImplementedError()

    @property
    def upload_as_links(self) -> bool:
        return self._upload_as_links

    @property
    def remote_files_map(self) -> Dict[str, str]:
        return self._remote_files_map

    @property
    def supports_links(self) -> bool:
        return self._supports_links

    def validate_labeling_interface(self) -> bool:
        return self._labeling_interface == LabelingInterface.DEFAULT

    def validate_ann_file(self, ann_path) -> bool:
        raise NotImplementedError()

    def validate_key_file(self) -> bool:
        raise NotImplementedError()

    def detect_format(self) -> BaseConverter:
        self._converter = self._detect_format()
        return self._converter

    def validate_format(self) -> bool:
        """
        Validate format of the input data meets the requirements of the converter. Should be implemented in the subclass.
        Additionally, this method must do the following steps:
            1. creates project meta (if key file file exists) and save it to self._meta
            2. creates items, count detected annotations and save them to self._items
            3. validates annotation files (and genereate meta if key file is missing)

        :return: True if format is valid, False otherwise.
        """
        raise NotImplementedError()

    def get_meta(self) -> ProjectMeta:
        return self._meta

    def get_items(self) -> List[BaseItem]:
        return self._items

    def to_supervisely(self, item: BaseItem, meta: ProjectMeta, *args) -> Annotation:
        """Convert to Supervisely format."""
        return item.create_empty_annotation()

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 50) -> None:
        """Upload converted data to Supervisely"""
        raise NotImplementedError()

    def _detect_format(self):
        found_formats = []
        all_converters = self.__class__.__subclasses__()

        progress, progress_cb = self.get_progress(1, "Detecting annotation format")
        for converter in all_converters:
            if converter.__name__ == "BaseConverter":
                continue
            converter = converter(
                self._input_data,
                self._labeling_interface,
                self._upload_as_links,
                self._remote_files_map,
            )

            if not converter.validate_labeling_interface():
                continue

            if self.upload_as_links and not converter.supports_links:
                continue

            if converter.validate_format():
                logger.info(f"Detected format: {str(converter)}")
                found_formats.append(converter)
                if len(found_formats) > 1:
                    raise RuntimeError(
                        f"Multiple formats detected: {[str(f) for f in found_formats]}. "
                        "Mixed formats are not supported yet."
                    )

        progress_cb(1)

        if len(found_formats) == 0:
            self._items, only_modality_items, unsupported_exts = (
                self._collect_items_if_format_not_detected()
            )

            if self.items_count == 0:
                if unsupported_exts:
                    raise RuntimeError(
                        f"Not found any {self.modality} to upload. "  # pylint: disable=no-member
                        f"Unsupported file extensions detected: {unsupported_exts}. "
                        f"Convert your data to one of the supported formats: {self.allowed_exts}"
                    )
                raise RuntimeError(
                    "Please refer to the app overview and documentation for annotation formats, "
                    "and ensure that your data contains valid information"
                )
            if not only_modality_items:
                logger.warn(
                    "Annotations not found. "  # pylint: disable=no-member
                    f"Uploading {self.modality} without annotations. "
                    "If you need assistance to upload data with annotations, please contact our support team."
                )
            return self

        if len(found_formats) == 1:
            return found_formats[0]

    def _collect_items_if_format_not_detected(self):
        only_modality_items = True
        unsupported_exts = set()
        items = []
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if ext.lower() in self.allowed_exts:  # pylint: disable=no-member
                    items.append(self.Item(full_path))  # pylint: disable=no-member
                    continue
                only_modality_items = False
                if ext.lower() in self.unsupported_exts:
                    unsupported_exts.add(ext)

        return items, only_modality_items, unsupported_exts

    def merge_metas_with_conflicts(
        self, api: Api, dataset_id: int
    ) -> Tuple[ProjectMeta, dict, dict]:

        # get meta1 from project and meta2 from converter
        dataset = api.dataset.get_info_by_id(dataset_id)
        if dataset is None:
            raise RuntimeError(
                f"Dataset ID:{dataset_id} not found. "
                "Please check if the dataset exists and try again."
            )
        meta1_json = api.project.get_meta(dataset.project_id, with_settings=True)
        meta1 = ProjectMeta.from_json(meta1_json)
        meta2 = self._meta

        if meta2 is None:
            return meta1, {}, {}

        # merge classes and tags from meta1 (unchanged) and meta2 (renamed if conflict)
        renamed_classes = {}
        renamed_tags = {}
        for new_cls in meta2.obj_classes:
            i = 1
            new_name = new_cls.name
            matched = False
            def _is_matched(old_cls: ObjClass, new_cls: ObjClass) -> bool:
                if old_cls.geometry_type == new_cls.geometry_type:
                    if old_cls.geometry_type == GraphNodes:
                        old_nodes = old_cls.geometry_config["nodes"]
                        new_nodes = new_cls.geometry_config["nodes"]
                        return old_nodes.keys() == new_nodes.keys()
                    return True
                return False

            while meta1.obj_classes.get(new_name) is not None:
                if _is_matched(meta1.get_obj_class(new_name), new_cls):
                    matched = True
                    break
                new_name = f"{new_cls.name}_{i}"
                i += 1
            if new_name != new_cls.name:
                logger.warn(f"Class {new_cls.name} renamed to {new_name}")
                renamed_classes[new_cls.name] = new_name
            if not matched:
                new_cls = new_cls.clone(name=new_name)
                meta1 = meta1.add_obj_class(new_cls)

        for new_tag in meta2.tag_metas:
            i = 1
            new_name = new_tag.name
            matched = False
            while meta1.tag_metas.get(new_name) is not None:
                if meta1.tag_metas.get(new_name).value_type == new_tag.value_type:
                    if new_tag.value_type != TagValueType.ONEOF_STRING:
                        matched = True
                        break
                    if meta1.tag_metas.get(new_name).possible_values == new_tag.possible_values:
                        matched = True
                        break
                new_name = f"{new_tag.name}_{i}"
                i += 1
            if new_name != new_tag.name:
                logger.warn(f"Tag {new_tag.name} renamed to {new_name}")
                renamed_tags[new_tag.name] = new_name
            if not matched:
                new_tag = new_tag.clone(name=new_name)
                meta1 = meta1.add_tag_meta(new_tag)

        meta1 = self._update_labeling_interface(meta1, meta2, renamed_tags)

        # update project meta
        meta1 = api.project.update_meta(dataset.project_id, meta1)

        return meta1, renamed_classes, renamed_tags

    def get_progress(
        self,
        total: int,
        message: str = "Processing items...",
        is_size: bool = False,
    ) -> tuple:
        if is_production():
            progress = Progress(message, total, is_size=is_size)
            progress_cb = progress.iters_done_report
        else:
            progress = tqdm(
                total=total, desc=message, unit="B" if is_size else "it", unit_scale=is_size
            )
            progress_cb = progress.update
        return progress, progress_cb

    def _update_labeling_interface(
        self,
        meta1: ProjectMeta,
        meta2: ProjectMeta,
        renamed_tags: Dict[str, str] = None,
    ) -> ProjectMeta:
        """
        Update project meta with labeling interface from the converter meta.
        Only update if the existing labeling interface is the default value.
        In other cases, the existing labeling interface is preserved.
        """
        existing = meta1.project_settings.labeling_interface
        new = meta2.project_settings.labeling_interface
        if existing == new or new == LabelingInterface.DEFAULT:
            return meta1

        group_tag_name = meta2.project_settings.multiview_tag_name
        if group_tag_name:
            group_tag_name = renamed_tags.get(group_tag_name, group_tag_name)
            new_settings = meta2.project_settings.clone(multiview_tag_name=group_tag_name)
        else:
            new_settings = meta2.project_settings

        return meta1.clone(project_settings=new_settings)

    def _download_remote_ann_files(self) -> None:
        """
        Download all annotation files from Cloud Storage to the local storage.
        Needed to detect annotation format if "upload_as_links" is enabled.
        """
        if not self.upload_as_links:
            return

        ann_archives = {l: r for l, r in self._remote_files_map.items() if is_archive(l)}

        anns_to_download = {
            l: r for l, r in self._remote_files_map.items() if get_file_ext(l) == self.ann_ext
        }
        if not anns_to_download and not ann_archives:
            return

        import asyncio

        for files_type, files in {
            "annotations": anns_to_download,
            "archives": ann_archives,
        }.items():
            if not files:
                continue

            is_archive_type = files_type == "archives"

            file_size = None
            if is_archive_type:
                logger.info(f"Remote archives detected.")
                file_size = sum(
                    self._api.storage.get_info_by_path(self._team_id, remote_path).sizeb
                    for remote_path in files.values()
                )

            loop = get_or_create_event_loop()
            _, progress_cb = self.get_progress(
                len(files) if not is_archive_type else file_size,
                f"Downloading {files_type} from remote storage",
                is_size=is_archive_type,
            )

            for local_path in files.keys():
                silent_remove(local_path)

            logger.info(f"Downloading {files_type} from remote storage...")
            download_coro = self._api.storage.download_bulk_async(
                team_id=self._team_id,
                remote_paths=list(files.values()),
                local_save_paths=list(files.keys()),
                progress_cb=progress_cb,
                progress_cb_type="number" if not is_archive_type else "size",
            )

            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(download_coro, loop=loop)
                future.result()
            else:
                loop.run_until_complete(download_coro)
            logger.info("Possible annotations downloaded successfully.")

            if is_archive_type:
                for local_path in files.keys():
                    parent_dir = Path(local_path).parent
                    if parent_dir.name == "ann":
                        target_dir = parent_dir
                    else:
                        target_dir = parent_dir / "ann"
                        target_dir.mkdir(parents=True, exist_ok=True)

                    unpack_archive(local_path, str(target_dir))
                    silent_remove(local_path)

                    dirs = [d for d in target_dir.iterdir() if d.is_dir()]
                    files = [f for f in target_dir.iterdir() if f.is_file()]
                    if len(dirs) == 1 and len(files) == 0:
                        for file in dirs[0].iterdir():
                            file.rename(target_dir / file.name)
                        remove_dir(str(dirs[0]))

                    logger.info(f"Archive {local_path} unpacked successfully to {str(target_dir)}")
