import os
import shutil
import numpy as np
from collections import OrderedDict
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from cachetools import LRUCache, Cache
from threading import Lock
from fastapi import Request

import supervisely as sly
from pathlib import Path


class PersistentImageLRUCache(LRUCache):
    __marker = object()

    def __init__(self, maxsize, filepath: Path, exist_ok: bool = False, getsizeof=None):
        super().__init__(maxsize)
        self._base_dir = filepath
        self._exist_ok = exist_ok

    def __getitem__(self, key: Any) -> Any:
        filepath = super(PersistentImageLRUCache, self).__getitem__(key)
        return sly.image.read(str(filepath))

    def __setitem__(self, key: Any, value: Any) -> None:
        if not self._base_dir.exists():
            self._base_dir.mkdir()

        filepath = self._base_dir / f"{str(key)}.png"
        super(PersistentImageLRUCache, self).__setitem__(key, filepath)

        if filepath.exists():
            sly.logger.debug(f"Rewrite image {str(filepath)}")
        sly.image.write(str(filepath), value)

    def pop(self, key, default=__marker):
        if key in self:
            filepath = super(PersistentImageLRUCache, self).__getitem__(key)
            value = self[key]
            del self[key]
            os.remove(filepath)
            sly.logger.debug(f"Remove {filepath} frame")
        elif default is self.__marker:
            raise KeyError(key)
        else:
            value = default
        return value

    def clear(self, rm_base_folder=True) -> None:
        while self.currsize > 0:
            self.popitem()
        if rm_base_folder:
            shutil.rmtree(self._base_dir)


def cache_pop_notifier(storage: list):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            value = func(*args, **kwargs)
            if "key" in kwargs:
                key = kwargs["key"]
            else:
                key = args[0]
            storage.append(key)
            return value

        return wrapper

    return decorator


class InferenceVideoCache:
    def __init__(
        self,
        app: sly.Application,
        maxsize_frames: int,
        is_persistant: bool = True,
        base_folder_in_app_data: str = "inference_cache",
        max_number_of_videos: Optional[int] = None,
    ) -> None:
        self._app = app
        self._is_persistant = is_persistant
        self._maxsize = maxsize_frames
        self._max_videos = max_number_of_videos
        self._cache = OrderedDict()

        if is_persistant:
            self._base_cls = PersistentImageLRUCache
            self._data_dir = Path(sly.app.get_data_dir()) / base_folder_in_app_data
            self._data_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._base_cls = LRUCache

    def get_or_create_cache_for_video(self, video_id: int) -> Type[Cache]:
        if video_id not in self._cache:
            self._check_max_num_videos_limits()
            if self._is_persistent:
                path = self._data_dir / str(video_id)
                cache_params = {"maxsize": self._maxsize, "filepath": path}
            else:
                cache_params = {"maxsize": self._maxsize}
            self._cache[video_id] = self._base_cls(**cache_params)

        return self._cache[video_id]

    def add_frames_to_cache(
        self,
        api: sly.Api,
        video_id: int,
        frame_index: Union[List[int], int],
    ):
        cache = self.get_or_create_cache_for_video(video_id)
        self.__update(video_id)
        self._add_to_cache(api, video_id, frame_index, cache)

    def clear_cache(self, video_id: int, rm_video_folder: bool = True):
        cache = self.get_or_create_cache_for_video(video_id)
        if isinstance(cache, PersistentImageLRUCache):
            cache.clear(rm_base_folder=rm_video_folder)
        else:
            cache.clear()
        del self._cache[video_id]

    def download_np(self, api: sly.Api, video_id: int, frame_index: int) -> np.ndarray:
        self.__update(video_id)
        cache = self.get_or_create_cache_for_video(video_id)
        if frame_index not in cache:
            self._add_to_cache(api, video_id, frame_index, cache)
        return cache[frame_index]

    def download_nps(
        self, api: sly.Api, video_id: int, frame_indexes: List[int]
    ) -> List[np.ndarray]:
        self.__update(video_id)
        cache = self.get_or_create_cache_for_video(video_id)
        indexes_to_load = []

        for fi in frame_indexes:
            if fi not in cache:
                indexes_to_load.append(fi)

        if len(indexes_to_load) > 0:
            self._add_to_cache(api, video_id, indexes_to_load, cache)

        return [cache[fi] for fi in frame_indexes]

    def add_cache_endpoint(self):
        server = self._app.get_server()

        @server.post("/cache_enpoint")
        def cache_endpoint(request: Request):
            # some cache stuff
            api: sly.Api = request.state.api
            state: dict = request.state.state
            api.logger.debug("Request state in cache endpoint", extra=state)
            video_id, frame_ranges = self._parse_state(state)
            for frange in frame_ranges:
                self.add_frames_to_cache(api, video_id, frange)

    def _parse_state(self, state: dict) -> Tuple[int, List[List[int]]]:
        """Get `video_id` and `frames_range` from state"""
        return 1, [[0, 1, 2, 3], [10, 11]]

    def _add_to_cache(
        self, api: sly.Api, video_id: int, frame_index: Union[int, List[int]], cache: Type[Cache]
    ):
        if isinstance(frame_index, int):
            frame_index = [frame_index]

        frames = api.video.frame.download_nps(video_id, frame_index)
        for fi, frame in zip(frame_index, frames):
            cache[fi] = frame

    def _check_max_num_videos_limits(self):
        if self._max_videos is None:
            return
        else:
            if len(self._cache) + 1 > self._max_videos:
                video_id = next(iter(self._cache))
                self.clear_cache(video_id, rm_video_folder=True)

    def __update(self, video_id: int):
        self._cache.move_to_end(video_id)
