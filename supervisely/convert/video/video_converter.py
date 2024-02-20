import cv2
from supervisely.io.json import dump_json_file
from supervisely.io.fs import file_exists
from supervisely import VideoAnnotation, Api, batched, logger, ProjectMeta
from supervisely.convert.base_converter import BaseConverter


class VideoConverter:
    class Item(BaseConverter.BaseItem):
        def __init__(self, item_path, ann_data=None, shape=None, custom_data={}, frame_count=None):
            self._path = item_path
            self._ann_data = ann_data
            self._type = "image"
            self._frame_count = None
            if shape is None:
                vcap = cv2.VideoCapture(item_path)
                if vcap.isOpened(): 
                    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self._shape = (height, width)
                    self._frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                self._shape = shape
                self._frame_count = frame_count
            self._custom_data = custom_data

        def create_empty_annotation(self):
            return VideoAnnotation(self._shape)

    def __init__(self, input_data):
        self._input_data = input_data
        self._meta = None
        self._items = []
        self._key_id_map = None
        self._converter = self._detect_format()

    @property
    def format(self):
        return self.converter.format

    @property
    def ann_ext(self):
        return None

    @property
    def key_file_ext(self):
        return None

    @staticmethod
    def validate_ann_file(ann_path, meta=None):
        return False

    def _detect_format(self):
        found_formats = []
        all_converters = VideoConverter.__subclasses__()
        for converter in all_converters:
            if converter.__name__ == "BaseConverter":
                continue
            converter = converter(self._input_data)
            if converter.validate_format():
                if len(found_formats) > 1:
                    raise RuntimeError(
                        f"Multiple formats detected: {found_formats}. "
                        "Mixed formats are not supported yet."
                    )
                found_formats.append(converter)

        if len(found_formats) == 0:
            logger.info(f"No valid dataset formats detected. Only image will be processed")
            return self  # TODO: list items if no valid format detected

        if len(found_formats) == 1:
            return found_formats[0]


    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 50):
        """Upload converted data to Supervisely"""

        dataset = api.dataset.get_info_by_id(dataset_id)
        meta_json = api.project.get_meta(dataset.project_id)
        meta = ProjectMeta.from_json(meta_json)
        meta = meta.merge(self._meta)

        api.project.update_meta(dataset.project_id, meta)

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            ann_paths = []
            for item in batch:
                item_names.append(item.name)
                item_paths.append(item.path)

                ann = self.to_supervisely(item, meta)
                ann_path = item.path + ".json"
                i = 0
                while file_exists(ann_path):
                    ann_path = item.path + f"_{i}.json"
                dump_json_file(ann.to_json(), ann_path)
                ann_paths.append(ann_path)

            img_infos = api.video.upload_paths(dataset_id, item_names, item_paths)
            img_ids = [img_info.id for img_info in img_infos]
            api.video.annotation.upload_paths(img_ids, ann_paths)

        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")