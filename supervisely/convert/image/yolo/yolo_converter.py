from supervisely.convert.base_format import AvailableImageFormats, BaseFormat


class YOLOFormat(BaseFormat):
    def __init__(self, input_data):
        super().__init__(input_data)

    def __str__(self):
        return AvailableImageFormats.YOLO

    def get_meta(self):
        raise NotImplementedError()
    
    def get_items(self):
        raise NotImplementedError()
    
    def to_supervisely(self):
        raise NotImplementedError()
