import cv2
import magic
import subprocess
import os

from tqdm import tqdm
from typing import List

from supervisely import Api, batched, generate_free_name, KeyIdMap, logger, is_development, Progress, ProjectMeta, VideoAnnotation
from supervisely.convert.base_converter import BaseConverter
from supervisely.video.video import ALLOWED_VIDEO_EXTENSIONS, get_info
from supervisely.io.fs import get_file_ext, get_file_name

class VideoConverter(BaseConverter):
    allowed_exts = ALLOWED_VIDEO_EXTENSIONS
    base_video_extension = ".mp4"

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

    def __init__(self, input_data: str, labeling_interface: str):
        self._input_data: str = input_data
        self._meta: ProjectMeta = None
        self._items: List[self.Item] = []
        self._key_id_map: KeyIdMap = None
        self._labeling_interface: str = labeling_interface
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

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        existing_names = set([vid.name for vid in api.video.get_list(dataset_id)])

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading videos...")
        else:
            progress_cb = None

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            anns = []
            figures_cnt = 0
            for item in batch:
                item_name, item_path = self.convert_to_mp4_if_needed(item.path)
                item_name = generate_free_name(existing_names, item_name, with_ext=True, extend_used_names=True)
                item.set_name(item_name)
                item.set_path(item_path)
                item_paths.append(item_path)
                item_names.append(item_name)

                ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
                figures_cnt += len(ann.figures)
                anns.append(ann)

            vid_infos = api.video.upload_paths(
                dataset_id,
                item_names,
                item_paths,
                item_progress=progress_cb,
            )
            vid_ids = [vid_info.id for vid_info in vid_infos]

            if log_progress:
                ann_progress = tqdm(total=figures_cnt, desc=f"Uploading annotations...")
                ann_progress_cb = ann_progress.update
            else:
                ann_progress_cb = None

            for video_id, ann in zip(vid_ids, anns):
                if ann is None:
                    ann = VideoAnnotation(item.shape, item.frame_count)
                api.video.annotation.append(video_id, ann, progress_cb=ann_progress_cb)

        if log_progress:
            if is_development():
                progress.close()
            ann_progress.close()
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")


    def convert_to_mp4_if_needed(self, video_path):
        video_name = get_file_name(video_path)
        # convert
        convert_progress = Progress(message=f"Converting {video_name}", total_cnt=1)
        output_video_name = f"{get_file_name(video_name)}{self.base_video_extension}"
        output_video_path = os.path.splitext(video_path)[0] + "_h264" + self.base_video_extension

        # check if video is already in mp4 format and mime type is `video/mp4`
        if video_path.lower().endswith(".mp4"):
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(video_path)
            if mime_type == "video/mp4":
                logger.info(
                    f'Video "{video_name}" is already in mp4 format, conversion is not required.'
                )
                return output_video_name, video_path

        # read video meta_data
        try:
            vid_meta = get_info(video_path)
            need_video_transc, need_audio_transc = self.check_codecs(vid_meta)
        except:
            need_video_transc, need_audio_transc = True, True

        # convert videos
        self.convert(
            input_path=video_path,
            output_path=output_video_path,
            need_video_transc=need_video_transc,
            need_audio_transc=need_audio_transc,
        )
        convert_progress.iter_done_report()

        return output_video_name, output_video_path


    def check_codecs(self, video_meta):
        need_video_transc, need_audio_transc = False, False
        for stream in video_meta["streams"]:
            codec_type = stream["codecType"]
            if codec_type not in ["video", "audio"]:
                continue
            codec_name = stream["codecName"]
            if codec_type == "video" and codec_name != "h264":
                logger.info(
                    f"Video codec is not h264, transcoding is required: {codec_name}"
                )
                need_video_transc = True
            elif codec_type == "audio" and codec_name != "aac":
                logger.info(
                    f"Audio codec is not aac, transcoding is required: {codec_name}"
                )
                need_audio_transc = True
        return need_video_transc, need_audio_transc


    def convert(self, input_path, output_path, need_video_transc, need_audio_transc):
        video_codec = "libx264" if need_video_transc else "copy"
        audio_codec = "aac" if need_audio_transc else "copy"
        logger.info("Converting video...")
        subprocess.call(
            [
                "ffmpeg",
                "-y",
                "-i",
                f"{input_path}",
                "-c:v",
                f"{video_codec}",
                "-c:a",
                f"{audio_codec}",
                f"{output_path}",
            ]
        )