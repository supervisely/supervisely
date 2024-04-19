import uuid
from collections import OrderedDict
from logging import Logger
from typing import Callable, Dict, Generator, List, Optional
from typing import OrderedDict as OrderedDictType

import numpy as np

import supervisely as sly
from supervisely.api.module_api import ApiField
from supervisely.geometry.geometry import Geometry
from supervisely.video_annotation.video_figure import VideoFigure


class TrackerInterface:
    def __init__(
        self,
        context,
        api,
        load_all_frames=False,
        notify_in_predict=False,
        per_point_polygon_tracking=True,
        frame_loader: Callable[[sly.Api, int, int], np.ndarray] = None,
        frames_loader: Callable[[sly.Api, int, List[int]], List[np.ndarray]] = None,
        should_notify: bool = True,
    ):
        self.api: sly.Api = api
        self.logger: Logger = api.logger
        self.frame_index = context["frameIndex"]
        self.frames_count = context["frames"]

        self.track_id = context["trackId"]
        self.video_id = context["videoId"]
        self.object_ids = list(context["objectIds"])
        self.figure_ids = list(context["figureIds"])
        self.direction = context["direction"]
        self.should_notify = should_notify

        # all geometries
        self.stop = len(self.figure_ids) * self.frames_count
        self.global_pos = 0
        self.global_stop_indicatior = False

        self.geometries: OrderedDictType[int, Geometry] = OrderedDict()
        self.frames_indexes: List[int] = []
        self._cur_frames_indexes: List[int] = []
        self._frames: Optional[np.ndarray] = None
        self.load_all_frames = load_all_frames
        self.per_point_polygon_tracking = per_point_polygon_tracking

        # increase self.stop by num of frames will be loaded
        self._add_frames_indexes()

        # increase self.stop by num of points
        self._add_geometries()

        self._hot_cache: Dict[int, np.ndarray] = {}
        self._local_cache_loader = frame_loader
        self._local_cache_frames_loader = frames_loader

        if self.load_all_frames:
            if notify_in_predict:
                self.stop += self.frames_count + 1
            self._load_frames_to_hot_cache()
            self._load_frames()

    def add_object_geometries(self, geometries: List[Geometry], object_id: int, start_fig: int):
        for frame, geometry in zip(self._cur_frames_indexes[1:], geometries):
            if self.global_stop_indicatior:
                self._notify(True, task="stop tracking")
                break
            self.add_object_geometry_on_frame(geometry, object_id, frame)

        self.geometries[start_fig] = geometries[-1]

    def frames_loader_generator(self, batch_size=16) -> Generator[None, None, None]:
        if self.load_all_frames:
            self._cur_frames_indexes = self.frames_indexes
            yield
            return

        self._load_frames_to_hot_cache(
            self.frames_indexes[: min(batch_size + 1, len(self.frames_indexes))]
        )
        ind = self.frames_indexes[0]
        frame = self._load_frame(ind)
        for next_ind_pos, next_ind in enumerate(self.frames_indexes[1:]):
            if next_ind not in self._hot_cache:
                self._load_frames_to_hot_cache(
                    self.frames_indexes[
                        next_ind_pos
                        + 1 : min(next_ind_pos + 1 + batch_size, len(self.frames_indexes))
                    ]
                )
            next_frame = self._load_frame(next_ind)
            self._frames = np.array([frame, next_frame])
            self.frames_count = 1
            self._cur_frames_indexes = [ind, next_ind]
            yield
            frame = next_frame
            ind = next_ind

            if self.global_stop_indicatior:
                self.clear_cache()
                return

    def add_object_geometries_on_frames(
        self,
        geometries: List[Geometry],
        object_ids: List[int],
        frame_indexes: List[int],
        notify: bool = True,
    ):
        def _split(geometries: List[Geometry], object_ids: List[int], frame_indexes: List[int]):
            result = {}
            for geometry, object_id, frame_index in zip(geometries, object_ids, frame_indexes):
                result.setdefault(object_id, []).append((geometry, frame_index))
            return result

        geometries_by_object = _split(geometries, object_ids, frame_indexes)

        for object_id, geometries_frame_indexes in geometries_by_object.items():
            figures_json = [
                {
                    ApiField.OBJECT_ID: object_id,
                    ApiField.GEOMETRY_TYPE: geometry.geometry_name(),
                    ApiField.GEOMETRY: geometry.to_json(),
                    ApiField.META: {ApiField.FRAME: frame_index},
                    ApiField.TRACK_ID: self.track_id,
                }
                for geometry, frame_index in geometries_frame_indexes
            ]
            figures_keys = [uuid.uuid4() for _ in figures_json]
            key_id_map = sly.KeyIdMap()
            self.api.video.figure._append_bulk(
                entity_id=self.video_id,
                figures_json=figures_json,
                figures_keys=figures_keys,
                key_id_map=key_id_map,
            )
            self.logger.debug(f"Added {len(figures_json)} geometries to object #{object_id}")
            if notify:
                self._notify(task="add geometry on frame", pos_increment=len(figures_json))

    def add_object_geometry_on_frame(
        self,
        geometry: Geometry,
        object_id: int,
        frame_ind: int,
        notify: bool = True,
    ):
        self.api.video.figure.create(
            self.video_id,
            object_id,
            frame_ind,
            geometry.to_json(),
            geometry.geometry_name(),
            self.track_id,
        )
        self.logger.debug(f"Added {geometry.geometry_name()} to frame #{frame_ind}")
        if notify:
            self._notify(task="add geometry on frame")

    def clear_cache(self):
        self._hot_cache.clear()

    def _add_geometries(self):
        self.logger.info("Adding geometries.")
        points = 0
        for figure_id in self.figure_ids:
            figure = self.api.video.figure.get_info_by_id(figure_id)
            geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
            self.geometries[figure_id] = geometry

            self.api.logger.debug(f"Added {figure.geometry_type} #{figure_id}")

            # per point track notification
            if isinstance(geometry, sly.Point):
                points += 1
            elif isinstance(geometry, sly.Polygon):
                points += len(geometry.exterior) + len(geometry.interior)
            elif isinstance(geometry, sly.GraphNodes):
                points += len(geometry.nodes.items())
            elif isinstance(geometry, sly.Polyline):
                points += len(geometry.exterior)

        if self.per_point_polygon_tracking:
            if not self.load_all_frames:
                self.stop += points * self.frames_count
            else:
                self.stop += points

        self.logger.info("Geometries added.")
        # TODO: other geometries

    def _add_frames_indexes(self):
        total_frames = self.api.video.get_info_by_id(self.video_id).frames_count
        cur_index = self.frame_index

        while 0 <= cur_index < total_frames and len(self.frames_indexes) < self.frames_count + 1:
            self.frames_indexes.append(cur_index)
            cur_index += 1 if self.direction == "forward" else -1

        if self.load_all_frames:
            self.stop += len(self.frames_indexes)

    def _load_frames_to_hot_cache(self, frames_indexes: List[int] = None):
        if self._local_cache_frames_loader is not None:
            if frames_indexes is None:
                frames_indexes = self.frames_indexes
            frames_to_load = []
            for frame_index in frames_indexes:
                if frame_index not in self._hot_cache:
                    frames_to_load.append(frame_index)
            if len(frames_to_load) == 0:
                return
            self.logger.info(f"Loading {frames_to_load} frames to hot cache.")
            loaded_rgbs = self._local_cache_frames_loader(self.api, self.video_id, frames_to_load)
            for rgb, loaded_frame_index in zip(loaded_rgbs, frames_to_load):
                self._hot_cache[loaded_frame_index] = rgb
            self.logger.info(f"{frames_to_load} frames loaded to hot cache.")

    def _load_frame(self, frame_index):
        if frame_index in self._hot_cache:
            return self._hot_cache[frame_index]
        if self._local_cache_loader is None:
            self._hot_cache[frame_index] = self.api.video.frame.download_np(
                self.video_id, frame_index
            )
            return self._hot_cache[frame_index]
        else:
            return self._local_cache_loader(self.api, self.video_id, frame_index)

    def _load_frames(self):
        rgbs = []
        self.logger.info(f"Loading {len(self.frames_indexes)} frames.")

        for frame_index in self.frames_indexes:
            img_rgb = self._load_frame(frame_index)
            rgbs.append(img_rgb)
            self._notify(task="load frame")
        self._frames = rgbs
        self.logger.info("Frames loaded.")

    def _notify(
        self,
        stop: bool = False,
        fstart: Optional[int] = None,
        fend: Optional[int] = None,
        task: str = "not defined",
        pos_increment: int = 1,
    ):
        if not self.should_notify:
            return

        self.global_pos += pos_increment

        if stop:
            pos = self.stop
        else:
            pos = self.global_pos

        fstart = min(self.frames_indexes) if fstart is None else fstart
        fend = max(self.frames_indexes) if fend is None else fend

        self.logger.debug(f"Task: {task}")
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

        if self.global_stop_indicatior and self.global_pos < self.stop:
            self.logger.info("Task stoped by user.")

    @property
    def frames(self) -> np.ndarray:
        return self._frames

    @property
    def frames_with_notification(self) -> np.ndarray:
        """Use this in prediction."""
        self._notify(task="get frames")
        return self._frames
