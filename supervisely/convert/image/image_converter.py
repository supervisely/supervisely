from supervisely.convert.base_format import BaseFormat


class ImageFormatConverter:
    # def __new__(cls) -> Self:
    #     pass
    converter = None

    def __init__(self, input_data):  # , output_format = AvailableFormats.SLY):
        self.converter = self.detect_format(input_data)  # -> converter class

    @property
    def format(self):
        return self.converter.format

    def detect_format(self, data) -> BaseFormat:
        """return converter class"""
        raise NotImplementedError()
