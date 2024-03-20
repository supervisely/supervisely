import xml.etree.ElementTree as ET
from typing import List

from supervisely import Annotation, ProjectMeta
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter


class PascalVOCConverter(ImageConverter):

    def __init__(self, input_data: str):
        self._input_data: str = input_data
        self._items: List[ImageConverter.Item] = []
        self._meta: ProjectMeta = None

    def __str__(self) -> str:
        return AvailableImageConverters.PASCAL_VOC

    @property
    def ann_ext(self) -> str:
        return ".xml"

    @property
    def key_file_ext(self) -> str:
        return None

    def validate_ann_format(ann_path) -> bool:
        tree = ET.parse(ann_path)
        root = tree.getroot()
        if root.tag == "annotation":
            return True
        return False

    def to_supervisely(
            self,
            item: ImageConverter.Item,
            meta: ProjectMeta = None,
            renamed_classes: dict = None,
            renamed_tags: dict = None,
    ) -> Annotation:
        raise NotImplementedError()

    def validate_format(self) -> bool:
        return False
