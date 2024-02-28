import os
from typing import List, Tuple, Union

from supervisely import Annotation, Api, ProjectMeta, logger, TagValueType
from supervisely.io.fs import JUNK_FILES, get_file_ext, get_file_name_with_ext


class AvailableImageConverters:
    SLY = "supervisely"
    COCO = "coco"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"


class AvailableVideoConverters:
    SLY = "supervisely"
    MOT = "mot"
    # DAVIS = "davis"


class AvailablePointcloudConverters:
    SLY = "supervisely"
    LAS = "las/laz"
    PLY = "ply"


class AvailablePointcloudEpisodesConverters:
    SLY = "supervisely"


class AvailableVolumeConverters:
    SLY = "supervisely"
    DICOM = "dicom"


class BaseConverter:
    class BaseItem:
        def __init__(
            self,
            item_path: str,
            ann_data: Union[str, dict] = None,
            shape: Union[Tuple, List] = None,
            custom_data: dict = {},
        ):
            self._path: str = item_path
            self._ann_data: Union[str, dict] = ann_data
            self._type: str = None
            self._shape: Union[Tuple, List] = shape
            self._custom_data: dict = custom_data

        @property
        def name(self) -> str:
            return get_file_name_with_ext(self._path)

        @property
        def path(self) -> str:
            return self._path

        @property
        def ann_data(self) -> Union[str, dict]:
            return self._ann_data

        @property
        def type(self) -> str:
            return self._type

        @property
        def shape(self) -> Union[Tuple, List]:
            return self._shape

        @property
        def custom_data(self) -> dict:
            return self._custom_data

        def set_path(self, path) -> None:
            self._path = path

        def set_ann_data(self, ann_data) -> None:
            self._ann_data = ann_data

        def set_shape(self, shape) -> None:
            self._shape = shape

        def set_custom_data(self, custom_data: dict) -> None:
            self.custom_data = custom_data

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
                self.set_path(item_path)
            if ann_data is not None:
                self.set_ann_data(ann_data)
            if shape is not None:
                self.set_shape(shape)
            if custom_data:
                self.update_custom_data(custom_data)

        def create_empty_annotation(self) -> Annotation:
            raise NotImplementedError()

    def __init__(self, input_data: str):
        self._input_data: str = input_data
        self._items: List[self.BaseItem] = []
        self._meta: ProjectMeta = None

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

    def to_supervisely(self, item: BaseItem, meta: ProjectMeta) -> Annotation:
        """Convert to Supervisely format."""
        return item.create_empty_annotation()

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 50) -> None:
        """Upload converted data to Supervisely"""
        raise NotImplementedError()

    def _detect_format(self):
        found_formats = []
        all_converters = self.__class__.__subclasses__()
        for converter in all_converters:
            if converter.__name__ == "BaseConverter":
                continue
            converter = converter(self._input_data)
            if converter.validate_format():
                found_formats.append(converter)
                if len(found_formats) > 1:
                    raise RuntimeError(
                        f"Multiple formats detected: {found_formats}. "
                        "Mixed formats are not supported yet."
                    )

        if len(found_formats) == 0:
            logger.info(f"No valid dataset formats detected. Only items will be processed")
            for root, _, files in os.walk(self._input_data):
                for file in files:
                    full_path = os.path.join(root, file)
                    ext = get_file_ext(full_path)
                    if file in JUNK_FILES:
                        continue
                    if ext in self.allowed_exts:
                        self._items.append(self.Item(full_path))
            return self

        if len(found_formats) == 1:
            return found_formats[0]

    def merge_metas_with_conflicts(
            self, meta1: ProjectMeta, meta2: ProjectMeta
    ) -> Tuple[ProjectMeta, dict, dict]:
        new_obj_classes = []
        renamed_classes = {}
        new_tags = []
        renamed_tags = {}
        for existing_cls in meta1.obj_classes:
            new_obj_classes.append(existing_cls)
        for new_cls in meta2.obj_classes:
            if meta1.obj_classes.get(new_cls.name) is not None:
                if meta1.obj_classes.get(new_cls.name).geometry_type == new_cls.geometry_type:
                    continue
            i = 1
            new_name = new_cls.name
            while meta1.obj_classes.get(new_name) is not None:
                new_name = f"{new_cls.name}_{i}"
                i += 1
            logger.warn(f"Class {new_cls.name} renamed to {new_name}")
            renamed_classes[new_cls.name] = new_name
            new_cls = new_cls.clone(name=new_name)
            new_obj_classes.append(new_cls)

        for existing_tag in meta1.tag_metas:
            new_tags.append(existing_tag)
        for new_tag in meta2.tag_metas:
            if meta1.tag_metas.get(new_tag.name) is not None:
                if meta1.tag_metas.get(new_tag.name).value_type == new_tag.value_type:
                    if new_tag.value_type != TagValueType.ONEOF_STRING:
                        continue
                    if meta1.tag_metas[new_tag.name].possible_values == new_tag.possible_values:
                        continue
            i = 1
            new_name = new_tag.name
            while meta1.tag_metas.get(new_name) is not None:
                new_name = f"{new_tag.name}_{i}"
                i += 1
            logger.warn(f"Tag {new_tag.name} renamed to {new_name}")
            renamed_tags[new_tag.name] = new_name
            new_tag = new_tag.clone(name=new_name)
            new_tags.append(new_tag)

        new_meta = meta1.clone(obj_classes=new_obj_classes, tag_metas=new_tags)
        return new_meta, renamed_classes, renamed_tags
