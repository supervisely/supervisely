from supervisely.convert.base_converter import AvailableVideoFormats, BaseConverter


class MOTConverter(BaseConverter):
    def __init__(self, input_data):
        super().__init__(input_data)

    def __str__(self):
        return AvailableVideoFormats.MOT

    def get_meta(self):
        return super().get_meta()

    def get_items(self):
        return super().get_items()

    def to_supervisely(self, image_path: str, ann_path: str):
        raise NotImplementedError()
