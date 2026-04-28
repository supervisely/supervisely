from supervisely.convert.base_converter import AvailableVideoConverters, BaseConverter


class DavisConverter(BaseConverter):
    """Base for DAVIS video segmentation format import (semi-supervised video object segmentation)."""

    def __str__(self):
        return AvailableVideoConverters.DAVIS

    def get_meta(self):
        return super().get_meta()

    def get_items(self):
        return super().get_items()

    def to_supervisely(self, image_path: str, ann_path: str):
        raise NotImplementedError()
