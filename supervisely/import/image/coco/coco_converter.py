class COCOFormat:
    def __init__(self):
        pass

    def to_supervisely(self, data, source_format, target_format):
        raise NotImplementedError()

    def from_supervisely(self, data, source_format, target_format):
        raise NotImplementedError()


input_dir = "path/to/dir"
