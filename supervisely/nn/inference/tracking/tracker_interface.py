import time
import uuid
from collections import OrderedDict, namedtuple
from logging import Logger
from queue import Queue
from threading import Lock, Thread
from typing import Any, Callable, Dict, Generator, List, Optional
from typing import OrderedDict as OrderedDictType

import numpy as np

from supervisely._utils import find_value_by_keys
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.graph import GraphNodes
from supervisely.geometry.helpers import deserialize_geometry
from supervisely.geometry.point import Point
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle
from supervisely.nn.inference.cache import InferenceImageCache
from supervisely.sly_logger import logger
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.annotation.label import LabelingStatus


class TrackerInterface:
    def __init__(
        self,
        context,
        api,
        load_all_frames=False,
        notify_in_predict=False,
        per_point_polygon_tracking=True,
        frame_loader: Callable[[Api, int, int], np.ndarray] = None,
        frames_loader: Callable[[Api, int, List[int]], List[np.ndarray]] = None,
        should_notify: bool = True,
    ):
        self.api: Api = api
        self.logger: Logger = api.logger
        self.frame_index = context["frameIndex"]
        self.frames_count = context["frames"]

        self.track_id = context["trackId"]
        self.video_id = context["videoId"]
        self.frame_width = context.get("frameWidth", None)
        self.frame_height = context.get("frameHeight", None)
        self._video_info = None
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

    @property
    def video_info(self):
        if self._video_info is None:
            self._video_info = self.api.video.get_info_by_id(self.video_id)
        return self._video_info

    def add_object_geometries(self, geometries: List[Geometry], object_id: int, start_fig: int):
        for frame, geometry in zip(self._cur_frames_indexes[1:], geometries):
            if self.global_stop_indicatior:
                self._notify(True, task="stop tracking")
                break
            self.add_object_geometry_on_frame(geometry, object_id, frame)

        self.geometries[start_fig] = geometries[-1]

    def frames_loader_generator(self, batch_size=4) -> Generator[None, None, None]:
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

    def _crop_geometry(self, geometry: Geometry):
        h, w = self.frame_height, self.frame_width
        if h is None or w is None:
            h = self.video_info.frame_height
            w = self.video_info.frame_width
        rect = Rectangle.from_size((h, w))
        cropped = geometry.crop(rect)
        if len(cropped) == 0:
            return None
        return cropped[0]

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
            for i, (geometry, frame_index) in enumerate(geometries_frame_indexes):
                geometries_frame_indexes[i] = (self._crop_geometry(geometry), frame_index)
            geometries_frame_indexes = [
                (geometry, frame_index)
                for geometry, frame_index in geometries_frame_indexes
                if geometry is not None
            ]
            figures_json = [
                {
                    ApiField.OBJECT_ID: object_id,
                    ApiField.GEOMETRY_TYPE: geometry.geometry_name(),
                    ApiField.GEOMETRY: geometry.to_json(),
                    ApiField.META: {ApiField.FRAME: frame_index},
                    ApiField.TRACK_ID: self.track_id,
                    ApiField.NN_CREATED: True,
                    ApiField.NN_UPDATED: True,
                }
                for geometry, frame_index in geometries_frame_indexes
            ]
            figures_keys = [uuid.uuid4() for _ in figures_json]
            key_id_map = KeyIdMap()
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
            status=LabelingStatus.AUTO,
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
            geometry = deserialize_geometry(figure.geometry_type, figure.geometry)
            self.geometries[figure_id] = geometry

            self.api.logger.debug(f"Added {figure.geometry_type} #{figure_id}")

            # per point track notification
            if isinstance(geometry, Point):
                points += 1
            elif isinstance(geometry, Polygon):
                points += len(geometry.exterior) + len(geometry.interior)
            elif isinstance(geometry, GraphNodes):
                points += len(geometry.nodes.items())
            elif isinstance(geometry, Polyline):
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


class ThreadSafeStopIndicator:
    def __init__(self):
        self._stopped = False
        self._reason = None
        self._lock = Lock()

    def stop(self, reason: Any = None):
        if self.is_stopped():
            return
        with self._lock:
            self._stopped = True
            self._reason = reason

    def is_stopped(self):
        with self._lock:
            return self._stopped

    def get_reason(self):
        with self._lock:
            return self._reason


FrameImage = namedtuple("FrameImage", ["frame_index", "image"])


class TrackerInterfaceV2:
    UPLOAD_SLEEP_TIME = 0.001  # 1ms
    NOTIFY_SLEEP_TIME = 1  # 1s

    def __init__(
        self,
        api: Api,
        context: Dict,
        cache: InferenceImageCache,
    ):
        self.api = api
        self.context = context
        self.video_id = find_value_by_keys(context, ["videoId", "video_id"])
        self.frame_index = find_value_by_keys(context, ["frameIndex", "frame_index"])
        self.frames_count = find_value_by_keys(context, ["frames", "framesCount", "frames_count"])
        self.track_id = context.get("trackId", "auto")
        self.direction = context.get("direction", "forward")
        self.session_id = find_value_by_keys(context, ["sessionId", "session_id"], None)
        self.figures = context.get("figures", None)
        self.direct_progress = context.get("useDirectProgressMessages", False)
        self.direction_n = 1 if self.direction == "forward" else -1
        self.stop_indicator = ThreadSafeStopIndicator()
        self.cache = cache
        self.frame_indexes = list(
            range(
                self.frame_index,
                self.frame_index + self.frames_count * self.direction_n + self.direction_n,
                self.direction_n,
            )
        )

        self.log_extra = {
            "video_id": self.video_id,
            "track_id": self.track_id,
            "session_id": self.session_id,
        }

        # start caching task
        self.run_cache_frames_task()

        self.upload_sleep_time = self.UPLOAD_SLEEP_TIME
        self.notify_sleep_time = self.NOTIFY_SLEEP_TIME
        self.upload_queue = Queue()
        self.notify_queue = Queue()
        self.upload_thread = None
        self.notify_thread = None
        self._upload_f = None
        self._notify_f = None

    def __call__(self, upload_f, notify_f):
        self.upload_f = upload_f
        self.notify_f = notify_f
        return self

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.wait_stop(exception=exc_value)
        return False

    @property
    def upload_f(self):
        return self._upload_f

    @upload_f.setter
    def upload_f(self, upload_f):
        self._upload_f = upload_f
        self.upload_thread = Thread(
            target=self._upload_loop,
            args=[
                self.upload_queue,
                self.notify_queue,
                self.stop_indicator,
                self._upload_f,
                self.upload_sleep_time,
                self._upload_exception_handler,
            ],
            daemon=True,
        )

    @property
    def notify_f(self):
        return self._notify_f

    @notify_f.setter
    def notify_f(self, notify_f):
        self._notify_f = notify_f
        self.notify_thread = Thread(
            target=self._nofify_loop,
            args=[
                self.notify_queue,
                self.stop_indicator,
                self._notify_f,
                self.notify_sleep_time,
                self._notify_exception_handler,
            ],
            daemon=True,
        )

    def start(self):
        if self.upload_thread is not None:
            self.upload_thread.start()
        if self.notify_thread is not None:
            self.notify_thread.start()

    def stop(self, exception: Exception = None):
        self.stop_indicator.stop(exception)

    def join(self, timeout: Optional[float] = 5 * 60):
        if self.upload_thread is not None and self.upload_thread.is_alive():
            self.upload_thread.join(timeout=timeout)
        if self.notify_thread is not None and self.notify_thread.is_alive():
            self.notify_thread.join(timeout=timeout)

    def wait_stop(self, exception: Exception = None, timeout: Optional[float] = 5 * 60):
        self.stop(exception)
        self.join(timeout=timeout)

    def is_stopped(self):
        return self.stop_indicator.is_stopped()

    def stop_reason(self):
        return self.stop_indicator.get_reason()

    @classmethod
    def _upload_loop(
        cls,
        q: Queue,
        notify_q: Queue,
        stop_indicator: ThreadSafeStopIndicator,
        upload_f: callable = None,
        upload_sleep_time: float = None,
        exception_handler: callable = None,
    ):
        logger.debug("Upload loop started")
        if upload_f is None:
            logger.warning("Upload function is not provided. Exiting upload loop.")
            return
        upload_sleep_time = upload_sleep_time or cls.UPLOAD_SLEEP_TIME
        try:
            while True:
                items = []
                while not q.empty():
                    items.append(q.get_nowait())
                if len(items) > 0:
                    upload_f(items)
                    for item in items:
                        notify_q.put(item)
                elif stop_indicator.is_stopped():
                    logger.debug("stop event is set. returning from upload loop")
                    return
                time.sleep(upload_sleep_time)
        except Exception as e:
            if exception_handler is not None:
                e = exception_handler(e)
                if not isinstance(e, Exception):
                    return
                raise e
            logger.error("Error in upload loop: %s", str(e), exc_info=True)
            raise

    @classmethod
    def _nofify_loop(
        cls,
        q: Queue,
        stop_indicator: ThreadSafeStopIndicator,
        notify_f: callable = None,
        notify_sleep_time: float = None,
        exception_handler: callable = None,
    ):
        logger.debug("Notify loop started")
        if notify_f is None:
            logger.warning("Notify function is not provided. Exiting notify loop.")
            return
        notify_sleep_time = notify_sleep_time or cls.NOTIFY_SLEEP_TIME
        try:
            while True:
                items = []
                while not q.empty():
                    items.append(q.get_nowait())
                if len(items) > 0:
                    notify_f(items)
                elif stop_indicator.is_stopped():
                    logger.debug(f"stop event is set. returning from notify loop")
                    return
                time.sleep(notify_sleep_time)
        except Exception as e:
            if exception_handler is not None:
                e = exception_handler(e)
                if not isinstance(e, Exception):
                    return
                raise e
            logger.error("Error in notify loop: %s", str(e), exc_info=True)
            raise

    def run_cache_frames_task(self):
        if self.cache.is_persistent:
            frame_ranges = None
        else:
            frame_ranges = [
                self.frame_index,
                self.frame_index + self.frames_count * self.direction_n,
            ]
            if self.direction_n == -1:
                frame_ranges = frame_ranges[::-1]
            frame_ranges = [frame_ranges]
        self.cache.run_cache_task_manually(
            self.api,
            frame_ranges,
            video_id=self.video_id,
        )

    def frames_loader_generator(
        self, batch_size=2, step=1
    ) -> Generator[List[FrameImage], None, None]:
        step = step * self.direction_n
        batch = []
        t = time.monotonic()
        for frame_i, frame in zip(
            self.frame_indexes,
            self.cache.frames_loader(
                self.api, video_id=self.video_id, frame_indexes=self.frame_indexes
            ),
        ):
            batch.append(FrameImage(frame_i, frame))
            if len(batch) == batch_size:
                _get_indexes_str = lambda b: ", ".join(map(lambda x: str(x.frame_index), b))
                if len(batch) > 16:
                    batch_indexes_str = (
                        _get_indexes_str(batch[:8]) + " ... " + _get_indexes_str(batch[-8:])
                    )
                else:
                    batch_indexes_str = _get_indexes_str(batch)

                logger.debug(
                    f"Frames [{batch_indexes_str}] loaded. Time: {time.monotonic() - t}",
                    extra=self.log_extra,
                )
                yield batch
                t = time.monotonic()
                batch = batch[step:]

    def load_all_frames(self):
        return next(self.frames_loader_generator(batch_size=len(self.frame_indexes)))

    def add_prediction(self, item: Any):
        self.upload_queue.put(item)

    def notify_progress(
        self,
        progress_current: int,
        progress_total: int,
        frame_range: List[int] = None,
    ):
        logger.debug(
            f"Notify progress: {progress_current}/{progress_total} on frames {frame_range}",
            extra=self.log_extra,
        )
        if self.direct_progress:
            self.api.vid_ann_tool.set_direct_tracking_progress(
                self.session_id,
                self.video_id,
                self.track_id,
                frame_range=frame_range,
                progress_current=progress_current,
                progress_total=progress_total,
            )
        else:
            stopped = self.api.video.notify_progress(
                self.track_id,
                self.video_id,
                frame_range[0],
                frame_range[1],
                progress_current,
                progress_total,
            )
            if stopped and progress_current < progress_total:
                logger.info("Task stopped by user.", extra=self.log_extra)
                self.stop()

    def notify_error(self, exception: Exception):
        logger.debug(f"Notify error: {str(exception)}", extra=self.log_extra)
        error = type(exception).__name__
        message = str(exception)
        if self.direct_progress:
            self.api.vid_ann_tool.set_direct_tracking_error(
                self.session_id,
                self.video_id,
                self.track_id,
                f"{type(exception).__name__}: {str(exception)}",
            )
        else:
            self.api.video.notify_tracking_error(self.track_id, error, message)

    def notify_warning(self, message: str):
        logger.debug(f"Notify warning: {message}", extra=self.log_extra)
        if self.direct_progress:
            self.api.vid_ann_tool.set_direct_tracking_warning(
                self.session_id, self.video_id, self.track_id, message
            )
        else:
            self.api.video.notify_tracking_warning(self.track_id, self.video_id, message)

    def _upload_exception_handler(self, exception: Exception):
        logger.error(
            "Error in upload loop: %s", str(exception), exc_info=True, extra=self.log_extra
        )
        self.stop_indicator.stop(exception)
        raise exception

    def _notify_exception_handler(self, exception: Exception):
        logger.error(
            "Error in notify loop: %s", str(exception), exc_info=True, extra=self.log_extra
        )
        self.stop_indicator.stop(exception)
        raise exception
