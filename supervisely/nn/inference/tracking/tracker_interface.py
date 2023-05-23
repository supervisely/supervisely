import numpy as np
from typing import Optional, List

import supervisely as sly
from supervisely.geometry.geometry import Geometry
from logging import Logger


class TrackerInterface:
    def __init__(self, context, api):
        self.api: sly.Api = api
        self.logger: Logger = api.logger
        self.frame_index = context["frameIndex"]
        self.frames_count = context["frames"]

        self.track_id = context["trackId"]
        self.video_id = context["videoId"]
        self.object_ids = list(context["objectIds"])
        self.figure_ids = list(context["figureIds"])
        self.direction = context["direction"]

        # all geometries
        self.stop = len(self.object_ids) * self.frames_count
        self.global_pos = 0
        self.global_stop_indicatior = False

        self.geometries: List[Geometry] = []
        self.frames_indexes: List[int] = []
        self._frames: Optional[np.ndarray] = None

        # inscrease self.stop by num of points
        self._add_geometries()
        # inscrease self.stop by num of frames will be loaded
        self._add_frames_indexes()
        self._load_frames()

    def add_object_geometries(self, geometries: List[Geometry], object_id: int):
        for frame, geometry in zip(self.frames_indexes[1:], geometries):
            self.api.video.figure.create(
                self.video_id,
                object_id,
                frame,
                geometry.to_json(),
                geometry.geometry_name(),
                self.track_id,
            )
            self._notify()

            if self.global_stop_indicatior:
                self.logger.info("Task stoped by user.")
                self._notify(True)
                break

    def _add_geometries(self):
        self.logger.info("Adding geometries.")
        for figure_id in self.figure_ids:
            figure = self.api.video.figure.get_info_by_id(figure_id)
            geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
            self.geometries.append(geometry)

            self.api.logger.debug(f"Added {figure.geometry_type} #{figure_id}")

            # per point track notification
            if isinstance(geometry, sly.Point):
                self.stop += 1
            elif isinstance(geometry, sly.Polygon):
                self.stop += len(geometry.exterior) + len(geometry.interior)
            elif isinstance(geometry, sly.GraphNodes):
                self.stop += len(geometry.nodes.items())

        self.logger.info("Geometries added.")
        # TODO: other geometries

    def _add_frames_indexes(self):
        total_frames = self.api.video.get_info_by_id(self.video_id).frames_count
        cur_index = self.frame_index

        while 0 <= cur_index < total_frames and len(self.frames_indexes) < self.frames_count + 1:
            self.frames_indexes.append(cur_index)
            cur_index += 1 if self.direction == "forward" else -1

        self.stop += len(self.frames_indexes)

    def _load_frames(self):
        rgbs = []
        self.logger.info(f"Loading {len(self.frames_indexes)} frames.")

        for frame_index in self.frames_indexes:
            img_rgb = self.api.video.frame.download_np(self.video_id, frame_index)
            rgbs.append(img_rgb)
            self._notify()
        self._frames = rgbs
        self.logger.info("Frames loaded.")

    def _notify(self, stop: bool = False):
        self.global_pos += 1
        if stop:
            pos = self.stop
        else:
            pos = self.global_pos

        self.logger.debug(f"Notification status: {pos}/{self.stop}")

        self.global_stop_indicatior = self.api.video.notify_progress(
            self.track_id,
            self.video_id,
            min(self.frames_indexes),
            max(self.frames_indexes),
            pos,
            self.stop,
        )

        self.logger.debug(f"Notification status: stop={self.global_stop_indicatior}")

    @property
    def frames(self) -> np.ndarray:
        return self._frames

    @property
    def frames_with_notification(self) -> np.ndarray:
        """Use this in prediction."""
        self._notify()
        return self._frames
