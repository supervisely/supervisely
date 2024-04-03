import cv2

from tqdm import tqdm
from typing import List

from supervisely import Api, batched, generate_free_name, KeyIdMap, logger, ProjectMeta, VideoAnnotation
from supervisely.convert.base_converter import BaseConverter
from supervisely.video.video import ALLOWED_VIDEO_EXTENSIONS

class VideoConverter(BaseConverter):
    allowed_exts = ALLOWED_VIDEO_EXTENSIONS

    class Item(BaseConverter.BaseItem):
        def __init__(
            self,
            item_path,
            ann_data=None,
            shape=None,
            custom_data=None,
            frame_count=None,
        ):
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
            self._custom_data = custom_data if custom_data is not None else {}

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
        return self._converter.format

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

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 1,
        log_progress=True,
    ):
        """Upload converted data to Supervisely"""

        dataset = api.dataset.get_info_by_id(dataset_id)
        existing_names = set([vid.name for vid in api.video.get_list(dataset.id)])
        if self._meta is not None:
            curr_meta = self._meta
        else:
            curr_meta = ProjectMeta()
        meta_json = api.project.get_meta(dataset.project_id)
        meta = ProjectMeta.from_json(meta_json)

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(meta, curr_meta)

        api.project.update_meta(dataset.project_id, meta)

        if log_progress:
            progress = tqdm(total=self.items_count, desc=f"Uploading videos...")
            progress_cb = progress.update
        else:
            progress_cb = None

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            anns = []
            figures_cnt = 0
            for item in batch:
                if item.name in existing_names:
                    new_name = generate_free_name(
                        existing_names, item.name, with_ext=True, extend_used_names=True
                    )
                    logger.warn(
                        f"Video with name '{item.name}' already exists, renaming to '{new_name}'"
                    )
                    item_names.append(new_name)
                else:
                    item_names.append(item.name)
                item_paths.append(item.path)

                ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
                figures_cnt += len(ann.figures)
                anns.append(ann)

            vid_infos = api.video.upload_paths(
                dataset_id,
                item_names,
                item_paths,
                progress_cb=progress_cb,
                item_progress=True,
            )
            vid_ids = [vid_info.id for vid_info in vid_infos]

            if log_progress:
                ann_progress = tqdm(total=figures_cnt, desc=f"Uploading annotations...")
                ann_progress_cb = ann_progress.update
            else:
                ann_progress_cb = None

            for video_id, ann in zip(vid_ids, anns):
                api.video.annotation.append(video_id, ann, progress_cb=ann_progress_cb)

        if log_progress:
            progress.close()
            ann_progress.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")
