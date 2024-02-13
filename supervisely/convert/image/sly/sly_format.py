from supervisely.convert.base_format import AvailableFormats, BaseFormat


class SLYFormat(BaseFormat):
    def __init__(self, input_data):
        super().__init__(input_data)

    def __str__(self):
        return AvailableFormats.SLY

    def to_supervisely(self):
        raise NotImplementedError()

    def to_coco(self):
        raise NotImplementedError()

    def to_pascal_voc(self):
        raise NotImplementedError()

    def to_yolo(self):
        raise NotImplementedError()
