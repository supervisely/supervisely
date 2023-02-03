import os
import shutil
import time

import cv2
import ffmpeg

import supervisely as sly
from tqdm import tqdm


class InferenceVideoInterface:
    def __init__(
        self, api, start_frame_index, frames_count, frames_direction, video_info, imgs_dir
    ):
        self.api = api

        self.video_info = video_info
        self.images_paths = []

        self.start_frame_index = start_frame_index
        self.frames_count = frames_count
        self.frames_direction = frames_direction

        self._video_fps = round(1 / self.video_info.frames_to_timecodes[1])

        self._geometries = []
        self._frames_indexes = []

        self._add_frames_indexes()

        self._frames_path = os.path.join(
            imgs_dir, f"video_inference_{video_info.id}_{time.time_ns()}", "frames"
        )
        self._imgs_dir = imgs_dir

        self._local_video_path = None

        os.makedirs(self._frames_path, exist_ok=True)

    def _add_frames_indexes(self):
        total_frames = self.video_info.frames_count
        cur_index = self.start_frame_index

        if self.frames_direction == "forward":
            end_point = (
                cur_index + self.frames_count
                if cur_index + self.frames_count < total_frames
                else total_frames
            )
            self._frames_indexes = [
                curr_frame_index for curr_frame_index in range(cur_index, end_point, 1)
            ]
        else:
            end_point = cur_index - self.frames_count if cur_index - self.frames_count > -1 else -1
            self._frames_indexes = [
                curr_frame_index for curr_frame_index in range(cur_index, end_point, -1)
            ]
            self._frames_indexes = []

    def _download_video_by_frames(self):
        for index, frame_index in tqdm(
            enumerate(self._frames_indexes),
            desc="Downloading frames",
            total=len(self._frames_indexes),
        ):
            frame_path = os.path.join(f"{self._frames_path}", f"frame{index:06d}.png")
            self.images_paths.append(frame_path)

            if os.path.isfile(frame_path):
                continue

            img_rgb = self.api.video.frame.download_np(self.video_info.id, frame_index)
            # save frame as PNG file
            sly.image.write(os.path.join(f"{self._frames_path}", f"frame{index:06d}.png"), img_rgb)

    def _download_entire_video(self):
        def videos_to_frames(video_path, frames_range=None):
            vidcap = cv2.VideoCapture(video_path)
            vidcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

            success, image = vidcap.read()
            count = 0

            while success:
                output_image_path = os.path.join(f"{self._frames_path}", f"frame{count:06d}.png")
                if frames_range:
                    if frames_range[0] <= count <= frames_range[1]:
                        cv2.imwrite(output_image_path, image)  # save frame as PNG file
                        self.images_paths.append(output_image_path)
                else:
                    cv2.imwrite(output_image_path, image)  # save frame as PNG file
                    self.images_paths.append(output_image_path)
                success, image = vidcap.read()
                count += 1

            fps = vidcap.get(cv2.CAP_PROP_FPS)

            return {"frames_path": self._frames_path, "fps": fps, "video_path": video_path}

        self._local_video_path = os.path.join(
            self._imgs_dir, f"{time.time_ns()}_{self.video_info.name}"
        )
        self.api.video.download_path(self.video_info.id, self._local_video_path)
        return videos_to_frames(
            self._local_video_path,
            [self.start_frame_index, self.start_frame_index + self.frames_count - 1],
        )

    def download_frames(self):
        if self.frames_count > (self.video_info.frames_count * 0.3):
            sly.logger.debug("Downloading entire video")
            self._download_entire_video()
        else:
            sly.logger.debug("Downloading video frame by frame")
            self._download_video_by_frames()

    def __del__(self):
        if os.path.isdir(self._frames_path):
            shutil.rmtree(os.path.dirname(self._frames_path), ignore_errors=True)

            if self._local_video_path is not None and os.path.isfile(self._local_video_path):
                os.remove(self._local_video_path)
