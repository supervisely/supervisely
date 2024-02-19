import supervisely.imaging.image as image
from supervisely import Annotation, logger
from supervisely.convert.base_converter import BaseConverter

ALLOWED_IMAGE_ANN_EXTENSIONS = [".json", ".txt", ".xml"]


class ImageConverter(BaseConverter):
    class Item(BaseConverter.BaseItem):
        def __init__(self, item_path, ann_data=None, shape=None, custom_data={}):
            self._path = item_path
            self._ann_data = ann_data
            self._type = "image"
            if shape is None:
                img = image.read(item_path)
                self._shape = img.shape[:2]
            else:
                self._shape = shape
            self._custom_data = custom_data

        def create_empty_annotation(self):
            if self._shape is not None:
                return Annotation(self._shape)
            else:
                ann = Annotation.from_img_path(self._path)
                self._shape = ann.img_size  # fill item shape
                return ann

    def __init__(self, input_data, items, annotations):
        self._input_data = input_data
        self._items = [self.Item(path) for path in items]
        self._annotations = annotations
        self._converter = self._detect_format()
        self._meta = None

    @property
    def format(self):
        return self._converter.format

    @property
    def items(self):
        return self._items

    @property
    def ann_ext(self):
        return None

    @property
    def key_file_ext(self):
        return None

    @staticmethod
    def validate_ann_file(ann_path):
        return False

    def require_key_file(self):
        return False

    def validate_key_files(self):
        return False

    def get_meta(self):
        return self._meta

    def _detect_format(self):
        found_formats = []
        all_converters = ImageConverter.__subclasses__()
        for converter in all_converters:
            converter = converter(self._input_data, self._items, self._annotations)
            if converter.validate_format():
                if len(found_formats) > 1:
                    raise RuntimeError(
                        f"Multiple formats detected: {found_formats}. "
                        "Mixed formats are not supported yet."
                    )
                found_formats.append(converter)

        if len(found_formats) == 0:
            logger.info(f"No valid dataset formats detected. Only image will be processed")
            return self

        if len(found_formats) == 1:
            return found_formats[0]
