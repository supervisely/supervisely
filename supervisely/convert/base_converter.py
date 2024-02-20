from typing import List, Tuple, Union

from supervisely import Annotation, Api, ProjectMeta
from supervisely.io.fs import get_file_name_with_ext


class AvailableImageConverters:
    SLY = "supervisely"
    COCO = "coco"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"


class AvailableVideoConverters:
    SLY = "supervisely"
    MOT = "mot"
    # DAVIS = "davis"


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

    @staticmethod
    def validate_ann_file(ann_path) -> bool:
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
        if item.ann_data is None:
            if item.shape is not None:
                return Annotation(item.shape)
            else:
                return Annotation.from_img_path(item.path)

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 50) -> None:
        """Upload converted data to Supervisely"""
        raise NotImplementedError()
