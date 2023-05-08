import numpy as np
from typing import Optional, List

import supervisely as sly
from supervisely.geometry.geometry import Geometry


class TrackerInterface:
    def __init__(self, context, api):
        self.api: sly.Api = api
        self.logger = api.logger
        self.frame_index = context["frameIndex"]
        self.frames_count = context["frames"]

        self.track_id = context["trackId"]
        self.video_id = context["videoId"]
        self.object_ids = list(context["objectIds"])
        self.figure_ids = list(context["figureIds"])
        self.direction = context["direction"]
        self.stop = len(self.object_ids) * self.frames_count
        self.global_pos = 0

        self.geometries: List[Geometry] = []
        self.frames_indexes: List[int] = []
        self.frames: Optional[np.ndarray] = None

        self._add_geometries()
        self._add_frames_indexes()
        self._load_frames()

    def add_object_geometries(self, geometries: List[Geometry], object_id: int):
        for frame, geometry in zip(self.frames_indexes, geometries):
            self.api.video.figure.create(
                self.video_id,
                object_id,
                frame,
                geometry.to_json(),
                geometry.geometry_name(),
                self.track_id,
            )
            stop = self._notify()
            self.global_pos += 1

            if stop:
                self.logger.info("Task stoped by user.")
                self._notify(True)
                break

    def _add_geometries(self):
        for figure_id in self.figure_ids:
            figure = self.api.video.figure.get_info_by_id(figure_id)
            geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
            self.geometries.append(geometry)

    def _add_frames_indexes(self):
        total_frames = self.api.video.get_info_by_id(self.video_id).frames_count
        cur_index = self.frame_index

        while 0 <= cur_index < total_frames and len(self.frames_indexes) < self.frames_count + 1:
            self.frames_indexes.append(cur_index)
            cur_index += 1 if self.direction == "forward" else -1

    def _load_frames(self):
        rgbs = []
        for frame_index in self.frames_indexes:
            img_rgb = self.api.video.frame.download_np(self.video_id, frame_index)
            rgbs.append(img_rgb)
        self.frames = rgbs

    def _notify(self, stop: bool = False):
        if stop:
            pos = self.stop
        else:
            pos = self.global_pos
        
        self.logger.debug(f"Notification status: {pos}/{self.stop}")

        nextstop = self.api.video.notify_progress(
            self.track_id,
            self.video_id,
            min(self.frames_indexes),
            max(self.frames_indexes),
            pos,
            self.stop,
        )

        self.logger.debug(f"Notification status: stop={nextstop}")

        return nextstop