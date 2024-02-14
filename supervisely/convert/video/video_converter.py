from supervisely.convert.base_format import BaseFormat


class VideoFormatConverter:
    converter = None

    def __init__(self, input_data):
        self.converter = self._detect_format(input_data)

    @property
    def format(self):
        return self.converter.format

    def _detect_format(self, data) -> BaseFormat:
        """return converter class"""
        raise NotImplementedError()
