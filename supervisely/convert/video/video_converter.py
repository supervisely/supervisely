from typing import List

import cv2

from supervisely import Api, KeyIdMap, ProjectMeta, VideoAnnotation, batched, logger
from supervisely.convert.base_converter import BaseConverter
from supervisely.io.fs import file_exists
from supervisely.io.json import dump_json_file


class VideoConverter:
    class Item(BaseConverter.BaseItem):
        def __init__(self, item_path, ann_data=None, shape=None, custom_data={}, frame_count=None):
            self._path = item_path
            self._ann_data = ann_data
            self._type = "video"
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

        @property
        def frame_count(self) -> int:
            return self._frame_count

        def create_empty_annotation(self) -> VideoAnnotation:
            return VideoAnnotation(self._shape, self._frame_count)

    def __init__(self, input_data: str):
        self._input_data: str = input_data
        self._meta: ProjectMeta = None
        self._items: List[self.Item] = []
        self._key_id_map: KeyIdMap = None
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

    def get_meta(self) -> ProjectMeta:
        return self._meta

    def get_items(self) -> List[BaseConverter.BaseItem]:
        return self._items

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

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1):
        """Upload converted data to Supervisely"""

        dataset = api.dataset.get_info_by_id(dataset_id)
        meta_json = api.project.get_meta(dataset.project_id)
        meta = ProjectMeta.from_json(meta_json)
        meta = meta.merge(self._meta)

        api.project.update_meta(dataset.project_id, meta)

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            anns = []
            for item in batch:
                item_names.append(item.name)
                item_paths.append(item.path)

                ann = self.to_supervisely(item, meta)
                anns.append(ann)

            vid_infos = api.video.upload_paths(dataset_id, item_names, item_paths)
            vid_ids = [vid_info.id for vid_info in vid_infos]
            for video_id, ann in zip(vid_ids, anns):
                api.video.annotation.append(video_id, ann)

        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")
