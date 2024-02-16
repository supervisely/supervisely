from supervisely.convert.base_converter import AvailableVideoFormats, BaseConverter


class SLYVideoConverter(BaseConverter):
    def __init__(self, input_data):
        super().__init__(input_data)

    def __str__(self):
        return AvailableVideoFormats.SLY

    def get_meta(self):
        return super().get_meta()

    def get_items(self):
        return super().get_items()

    def to_supervisely(self):
        raise NotImplementedError()

    def to_coco(self):
        raise NotImplementedError()

    def to_pascal_voc(self):
        raise NotImplementedError()

    def to_yolo(self):
        raise NotImplementedError()
