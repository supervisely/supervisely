class FormatConverter:
    def __init__(self):
        self.formats = {
            'yolo': YOLOFormat(),
            'coco': COCOFormat(),
            'supervisely': SuperviselyFormat()
        }

    def convert(self, data, source_format, target_format):
        if source_format not in self.formats or target_format not in self.formats:
            raise ValueError("Unsupported format")
        
        source_format_obj = self.formats[source_format]
        target_format_obj = self.formats[target_format]
        
        if source_format != 'supervisely':
            data = source_format_obj.to_supervisely(data)
        
        if target_format != 'supervisely':
            data = target_format_obj.from_supervisely(data)
        
        return data

