import numpy as np
from typing import Generator, Optional, List, Tuple, OrderedDict
from collections import OrderedDict

import supervisely as sly
from supervisely.geometry.geometry import Geometry
from logging import Logger


class TrackerInterface:
    def __init__(self, context, api, load_all_frames=False):
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

        self.geometries: OrderedDict[int, Geometry] = OrderedDict()
        self.frames_indexes: List[int] = []
        self._cur_frames_indexes: List[int] = []
        self._frames: Optional[np.ndarray] = None
        self.load_all_frames = load_all_frames

        # increase self.stop by num of frames will be loaded
        self._add_frames_indexes()

        # increase self.stop by num of points
        self._add_geometries()

        if self.load_all_frames:
            self._load_frames()

    def add_object_geometries(self, geometries: List[Geometry], object_id: int, start_fig: int):
        for frame, geometry in zip(self._cur_frames_indexes[1:], geometries):
            self.add_object_geometry_on_frame(geometry, object_id, frame)

            if self.global_stop_indicatior:
                self.logger.info("Task stoped by user.")
                self._notify(True)
                break

        self.geometries[start_fig] = geometries[-1]

    def frames_loader_generator(self) -> Generator[None, None, None]:
        if self.load_all_frames:
            self._cur_frames_indexes = self.frames_indexes
            yield
            return

        ind = self.frames_indexes[0]
        frame = self._load_frame(ind)
        for next_ind in self.frames_indexes[1:]:
            next_frame = self._load_frame(next_ind)
            self._frames = np.array([frame, next_frame])
            self.frames_count = 1
            self._cur_frames_indexes = [ind, next_ind]
            yield
            frame = next_frame
            ind = next_ind

            if self.global_stop_indicatior:
                return

    def add_object_geometry_on_frame(self, geometry: Geometry, object_id: int, frame_ind: int):
        self.api.video.figure.create(
            self.video_id,
            object_id,
            frame_ind,
            geometry.to_json(),
            geometry.geometry_name(),
            self.track_id,
        )
        self.logger.debug(f"Added {geometry.geometry_name()} to frame #{frame_ind}")
        self._notify(fstart=frame_ind, fend=frame_ind + 1)

    def _add_geometries(self):
        self.logger.info("Adding geometries.")
        for figure_id in self.figure_ids:
            figure = self.api.video.figure.get_info_by_id(figure_id)
            geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
            self.geometries[figure_id] = geometry

            self.api.logger.debug(f"Added {figure.geometry_type} #{figure_id}")

            # per point track notification
            if self.load_all_frames is not None:
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

    def _load_frame(self, frame_index):
        return self.api.video.frame.download_np(self.video_id, frame_index)

    def _load_frames(self):
        rgbs = []
        self.logger.info(f"Loading {len(self.frames_indexes)} frames.")

        for frame_index in self.frames_indexes:
            img_rgb = self._load_frame(frame_index)
            rgbs.append(img_rgb)
            self._notify()
        self._frames = rgbs
        self.logger.info("Frames loaded.")

    def _notify(
        self,
        stop: bool = False,
        fstart: Optional[int] = None,
        fend: Optional[int] = None,
    ):
        self.global_pos += 1
        if stop:
            pos = self.stop
        else:
            pos = self.global_pos

        fstart = min(self.frames_indexes) if fstart is None else fstart
        fend = max(self.frames_indexes) if fend is None else fend

        self.logger.debug(f"Notification status: {pos}/{self.stop}")

        self.global_stop_indicatior = self.api.video.notify_progress(
            self.track_id,
            self.video_id,
            fstart,
            fend,
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
