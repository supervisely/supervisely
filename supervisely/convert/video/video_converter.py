import os
import subprocess
from typing import Dict, Optional, Tuple, Union

import cv2
import magic

from supervisely import (
    Api,
    KeyIdMap,
    Progress,
    VideoAnnotation,
    batched,
    generate_free_name,
    is_development,
    logger,
)
from supervisely.convert.base_converter import BaseConverter
from supervisely.io.fs import (
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
    silent_remove,
)
from supervisely.project.project_settings import LabelingInterface
from supervisely.video.video import ALLOWED_VIDEO_EXTENSIONS, get_info


class VideoConverter(BaseConverter):
    allowed_exts = ALLOWED_VIDEO_EXTENSIONS + [".mpg"]
    base_video_extension = ".mp4"
    modality = "videos"

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
            self._name: str = None
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
        def shape(self) -> Tuple[int, int]:
            return self._shape

        @shape.setter
        def shape(self, shape: Optional[Tuple[int, int]] = None):
            self._shape = shape if shape is not None else [None, None]

        @property
        def frame_count(self) -> int:
            return self._frame_count

        @frame_count.setter
        def frame_count(self, frame_count: int):
            self._frame_count = frame_count

        @property
        def name(self) -> str:
            if self._name is not None:
                return self._name
            return get_file_name_with_ext(self._path)

        @name.setter
        def name(self, name: str):
            self._name = name

        def create_empty_annotation(self) -> VideoAnnotation:
            return VideoAnnotation(self._shape, self._frame_count)

    def __init__(
        self,
        input_data: str,
        labeling_interface: Optional[Union[LabelingInterface, str]],
        upload_as_links: bool,
        remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)
        self._key_id_map: KeyIdMap = None

    @property
    def format(self):
        return self._converter.format

    @property
    def ann_ext(self):
        return None

    @property
    def key_file_ext(self):
        return None

    @staticmethod
    def validate_ann_file(ann_path, meta=None):
        return False

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 10,
        log_progress=True,
    ):
        """Upload converted data to Supervisely"""

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        videos_in_dataset = api.video.get_list(dataset_id, force_metadata_for_links=False)
        existing_names = {video_info.name for video_info in videos_in_dataset}

        # check video codecs, mimetypes and convert if needed
        convert_progress, convert_progress_cb = self.get_progress(
            self.items_count, "Preparing videos..."
        )
        for item in self._items:
            item_name, item_path = self.convert_to_mp4_if_needed(item.path)
            item.name = item_name
            item.path = item_path
            convert_progress_cb(1)
        if is_development():
            convert_progress.close()

        has_large_files = False
        size_progress_cb = None
        progress_cb, progress, ann_progress, ann_progress_cb = None, None, None, None
        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading videos...")
            if not self.upload_as_links:
                file_sizes = [get_file_size(item.path) for item in self._items]
                has_large_files = any(
                    [self._check_video_file_size(file_size) for file_size in file_sizes]
                )
                if has_large_files:
                    upload_progress = []
                    size_progress_cb = self._get_video_upload_progress(upload_progress)
        batch_size = 1 if has_large_files and not self.upload_as_links else batch_size

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            anns = []
            figures_cnt = 0
            for item in batch:
                item.name = generate_free_name(
                    existing_names, item.name, with_ext=True, extend_used_names=True
                )
                item_paths.append(item.path)
                item_names.append(item.name)

                ann = None
                if not self.upload_as_links or self.supports_links:
                    ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
                    if ann is not None:
                        figures_cnt += len(ann.figures)
                anns.append(ann)

            if self.upload_as_links:
                vid_infos = api.video.upload_links(
                    dataset_id,
                    item_paths,
                    item_names,
                    skip_download=True,
                    progress_cb=progress_cb if log_progress else None,
                    force_metadata_for_links=False,
                )
            else:
                vid_infos = api.video.upload_paths(
                    dataset_id,
                    item_names,
                    item_paths,
                    progress_cb=progress_cb if log_progress else None,
                    item_progress=(size_progress_cb if log_progress and has_large_files else None),
                )
            vid_ids = [vid_info.id for vid_info in vid_infos]

            if log_progress and has_large_files and figures_cnt > 0:
                ann_progress, ann_progress_cb = self.get_progress(
                    figures_cnt, "Uploading annotations..."
                )

            for vid, ann, item, info in zip(vid_ids, anns, batch, vid_infos):
                if ann is None:
                    ann = VideoAnnotation((info.frame_height, info.frame_width), info.frames_count)
                api.video.annotation.append(vid, ann, progress_cb=ann_progress_cb)

        if log_progress and is_development():
            if progress is not None:
                progress.close()
            if ann_progress is not None:
                ann_progress.close()
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")

    def convert_to_mp4_if_needed(self, video_path):
        video_name = get_file_name_with_ext(video_path)
        if self.upload_as_links:
            video_path = self.remote_files_map.get(video_path)
            return video_name, video_path
        video_ext = get_file_ext(video_path)
        if video_ext.lower() != video_ext:
            # rename video file to make extension lowercase
            new_video_path = os.path.splitext(video_path)[0] + video_ext.lower()
            os.rename(video_path, new_video_path)
            video_path = new_video_path

        # convert
        output_video_name = f"{get_file_name(video_name)}{self.base_video_extension}"
        output_video_path = os.path.splitext(video_path)[0] + "_h264" + self.base_video_extension

        # read video meta_data
        try:
            vid_meta = get_info(video_path)
            need_video_transc, need_audio_transc = self._check_codecs(vid_meta)
        except:
            need_video_transc, need_audio_transc = True, True

        if not need_video_transc:
            # check if video is already in mp4 format and mime type is `video/mp4`
            if video_path.lower().endswith(".mp4"):
                mime = magic.Magic(mime=True)
                mime_type = mime.from_file(video_path)
                if mime_type == "video/mp4":
                    logger.debug(
                        f'Video "{video_name}" is already in mp4 format, conversion is not required.'
                    )
                    return output_video_name, video_path
                else:
                    need_video_transc = True
            else:
                need_video_transc = True

        # convert videos
        self._convert(
            input_path=video_path,
            output_path=output_video_path,
            need_video_transc=need_video_transc,
            need_audio_transc=need_audio_transc,
        )
        if os.path.exists(output_video_path):
            logger.info(f"Video {video_name} has been converted.")
            silent_remove(video_path)

        return output_video_name, output_video_path

    def _check_codecs(self, video_meta):
        need_video_transc, need_audio_transc = False, False
        for stream in video_meta["streams"]:
            codec_type = stream["codecType"]
            if codec_type not in ["video", "audio"]:
                continue
            codec_name = stream["codecName"]
            if codec_type == "video" and codec_name not in ["h264", "h265", "hevc", "av1"]:
                logger.info(f"Video codec is not h264/h265/hevc/av1, transcoding is required: {codec_name}")
                need_video_transc = True
            elif codec_type == "audio" and codec_name != "aac":
                logger.info(f"Audio codec is not aac, transcoding is required: {codec_name}")
                need_audio_transc = True
        return need_video_transc, need_audio_transc

    def _convert(self, input_path, output_path, need_video_transc, need_audio_transc):
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

    def _check_video_file_size(self, file_size):
        return file_size > 20 * 1024 * 1024  # 20 MB

    def _get_video_upload_progress(self, upload_progress):
        upload_progress = []

        def _print_progress(monitor, upload_progress):
            if len(upload_progress) == 0:
                upload_progress.append(
                    Progress(
                        message="Upload videos...",
                        ext_logger=logger,
                        is_size=True,
                    )
                )
            upload_progress[0].set_current_value(monitor)

        return lambda m: _print_progress(m, upload_progress)
