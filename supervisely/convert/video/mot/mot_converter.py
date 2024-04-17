from supervisely import Annotation, ProjectMeta
from supervisely.convert.base_converter import AvailableVideoConverters, BaseConverter
from supervisely.convert.video.video_converter import VideoConverter


class MOTConverter(VideoConverter):
    def __init__(self, input_data, labeling_interface: str):
        self._input_data: str = input_data
        self._labeling_interface: str = labeling_interface

    def __str__(self):
        return AvailableVideoConverters.MOT

    def validate_format(self):
        return False

    def to_supervisely(self, item: VideoConverter.Item, meta: ProjectMeta) -> Annotation:
        raise NotImplementedError()
