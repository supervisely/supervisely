from __future__ import annotations

import os
from typing import List, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from tqdm import tqdm

from supervisely._utils import is_production
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.tag_meta import TagValueType
from supervisely.api.api import Api
from supervisely.io.fs import JUNK_FILES, get_file_ext, get_file_name_with_ext
from supervisely.project.project_meta import ProjectMeta
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


class AvailableVideoConverters:
    SLY = "supervisely"
    MOT = "mot"
    DAVIS = "davis"


class AvailablePointcloudConverters:
    SLY = "supervisely"
    LAS = "las/laz"
    PLY = "ply"
    BAG = "rosbag"


class AvailablePointcloudEpisodesConverters:
    SLY = "supervisely"
    BAG = "rosbag"


class AvailableVolumeConverters:
    SLY = "supervisely"
    DICOM = "dicom"


class BaseConverter:
    unsupported_exts = [".gif", ".html", ".htm"]

    class BaseItem:
        def __init__(
            self,
            item_path: str,
            ann_data: Union[str, dict] = None,
            shape: Union[Tuple, List] = None,
            custom_data: dict = {},
        ):
            self._path: str = item_path
            self._name: str = None
            self._ann_data: Union[str, dict, list] = ann_data
            self._shape: Union[Tuple, List] = shape
            self._custom_data: dict = custom_data

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
        labeling_interface: Literal[
            "default",
            "multi_view",
            "multispectral",
            "images_with_16_color",
            "medical_imaging_single",
        ] = "default",
    ):
        self._input_data: str = input_data
        self._items: List[self.BaseItem] = []
        self._meta: ProjectMeta = None
        self._labeling_interface: str = labeling_interface

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

    def validate_labeling_interface(self) -> bool:
        return self._labeling_interface == "default"

    def validate_ann_file(self, ann_path) -> bool:
        raise NotImplementedError()

    def validate_key_file(self) -> bool:
        raise NotImplementedError()

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
            converter = converter(self._input_data, self._labeling_interface)
            if not converter.validate_labeling_interface():
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
            only_modality_items = True
            unsupported_exts = set()
            for root, _, files in os.walk(self._input_data):
                for file in files:
                    full_path = os.path.join(root, file)
                    ext = get_file_ext(full_path)
                    if ext.lower() in self.allowed_exts:  # pylint: disable=no-member
                        self._items.append(self.Item(full_path))  # pylint: disable=no-member
                        continue
                    only_modality_items = False
                    if ext.lower() in self.unsupported_exts:
                        unsupported_exts.add(ext)

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
            while meta1.obj_classes.get(new_name) is not None:
                if meta1.obj_classes.get(new_name).geometry_type == new_cls.geometry_type:
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
