import os
import shutil
import numpy as np
from typing import Any, List, Tuple, Union
from cachetools import LRUCache, Cache, TTLCache
from threading import Lock
from fastapi import Request
from enum import Enum

import supervisely as sly
from supervisely.io.fs import silent_remove
from pathlib import Path


class PersistentImageLRUCache(LRUCache):
    __marker = object()

    def __init__(self, maxsize, filepath: Path, getsizeof=None):
        super().__init__(maxsize)
        self._base_dir = filepath

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
            filepath = self._base_dir / f"{str(key)}.png"
            value = self[key]
            del self[key]
            silent_remove(filepath)
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


class PersistentImageTTLCache(TTLCache):
    def __init__(self, maxsize: int, ttl: int, filepath: Path):
        super().__init__(maxsize, ttl)
        # TTLCache.__init__(self, maxsize, ttl)
        self._base_dir = filepath

    def __getitem__(self, key: Any) -> np.ndarray:
        filepath = super(PersistentImageTTLCache, self).__getitem__(key)
        return sly.image.read(str(filepath))

    def __setitem__(self, key: Any, value: np.ndarray) -> None:
        if not self._base_dir.exists():
            self._base_dir.mkdir()

        filepath = self._base_dir / f"{str(key)}.png"
        super(PersistentImageTTLCache, self).__setitem__(key, filepath)

        if filepath.exists():
            sly.logger.debug(f"Rewrite image {str(filepath)}")
        sly.image.write(str(filepath), value)

    def __delitem__(self, key: Any) -> None:
        cache_delitem = PersistentImageTTLCache.__delitem
        return super().__delitem__(key, cache_delitem=cache_delitem)

    def __delitem(self, key: Any):
        Cache.__delitem__(self, key)
        filepath = self._base_dir / f"{str(key)}.png"
        silent_remove(filepath)

    def expire(self, time=None):
        """Remove expired items from the cache."""
        if time is None:
            time = self.timer()
        root = self._TTLCache__root
        curr = root.next
        links = self._TTLCache__links
        cache_delitem = PersistentImageTTLCache.__delitem
        while curr is not root and not (time < curr.expires):
            cache_delitem(self, curr.key)
            del links[curr.key]
            next = curr.next
            curr.unlink()
            curr = next

    def clear(self, rm_base_folder=True) -> None:
        while self.currsize > 0:
            self.popitem()
        if rm_base_folder:
            shutil.rmtree(self._base_dir)


# TODO: Add lock on cache changes
class SmartSegCache:
    class _ImageLoadType(Enum):
        ImageId: str = "IMAGE"
        ImageHash: str = "HASH"
        Frame: str = "FRAME"

    def __init__(
        self,
        app: sly.Application,
        maxsize: int,
        ttl: int,
        is_persistent: bool = True,
        base_folder: str = "/tmp/smart_tool_cache",
    ) -> None:
        self._app = app
        self._is_persistent = is_persistent
        self._maxsize = maxsize
        self._ttl = ttl

        if is_persistent:
            self._data_dir = Path(base_folder)
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._cache = PersistentImageTTLCache(maxsize, ttl, self._data_dir)
        else:
            self._cache = TTLCache(maxsize, ttl)

    def clear_cache(self):
        self._cache.clear(False)

    def download_image(self, api: sly.Api, image_id: int):
        name = f"image_{image_id}"
        if name not in self._cache:
            img = api.image.download_np(image_id)
            self._add_to_cache(name, img)
        return self._cache[name]

    def download_images(self, api: sly.Api, image_ids: List[int]):
        # TODO: bulk load with dataset_id
        return [self.download_image(api, imid) for imid in image_ids]

    def download_image_by_hash(self, api: sly.Api, img_hash: Any):
        true_name = self._image_name(img_hash)
        path = self._data_dir / f"tmp_{true_name}.png"
        api.image.download_paths_by_hashes([img_hash], [path])
        self._cache[true_name] = sly.image.read(path)
        silent_remove(path)

    # def download_images(self, api: sly.Api, dataset_id: int, image_ids: List[int]):
    #     images_to_load = []
    #     names_to_load = []

    #     for img_id in image_ids:
    #         name = self._image_name(img_id)
    #         if name not in self._cache:
    #             images_to_load.append(img_id)
    #             names_to_load.append(name)
    #         images = api.image.download_nps(dataset_id, images_to_load)

    #     self._add_to_cache(names_to_load, images)
    #     return [self._cache[self._image_name(img_id)] for img_id in image_ids]

    def download_frame(self, api: sly.Api, video_id: int, frame_index: int) -> np.ndarray:
        name = self._frame_name(video_id, frame_index)
        if name not in self._cache:
            frame = api.video.frame.download_np(video_id, frame_index)
            self._add_to_cache(name, frame)

        return self._cache[name]

    def download_frames(
        self,
        api: sly.Api,
        video_id: int,
        frame_indexes: List[int],
    ) -> List[np.ndarray]:
        indexes_to_load = []
        names_to_load = []

        for fi in frame_indexes:
            name = self._frame_name(video_id, fi)
            if fi not in self._cache:
                names_to_load.append(name)
                indexes_to_load.append(fi)

        if len(indexes_to_load) > 0:
            frames = api.video.frame.download_nps(video_id, indexes_to_load)
            self._add_to_cache(api, names_to_load, frames)

        return [self._cache[fi] for fi in frame_indexes]

    def add_cache_endpoint(self):
        server = self._app.get_server()

        # TODO: change endpoint name
        @server.post("/cache_enpoint")
        def cache_endpoint(request: Request):
            api: sly.Api = request.state.api
            state: dict = request.state.state
            api.logger.debug("Request state in cache endpoint", extra=state)
            image_ids, task_type = self._parse_state(state)

            if task_type is SmartSegCache._ImageLoadType.ImageId:
                self.download_images(api, image_ids)
            elif task_type is SmartSegCache._ImageLoadType.ImageHash:
                # TODO: add hashes load if needed
                self.download_image_by_hash(api, image_ids[0])
            elif task_type is SmartSegCache._ImageLoadType.Frame:
                video_id = state["video_id"]
                self.download_frames(api, video_id, image_ids)

    @property
    def ttl(self):
        return self._ttl

    @property
    def tmp_path(self):
        if self._is_persistent:
            return str(self._data_dir)
        return None

    def _parse_state(self, state: dict) -> Tuple[List[Any], _ImageLoadType]:
        if "image_ids" in state:
            return state["image_ids"], SmartSegCache._ImageLoadType.ImageId
        elif "image_hashes" in state:
            return state["image_hashes"], SmartSegCache._ImageLoadType.ImageHash
        elif "video_id" in state:
            frame_ranges = state["frame_ranges"]
            frames = []
            for fr_range in frame_ranges:
                start, end = fr_range[0], fr_range[1] + 1
                frames.extend(list(range(start, end)))
            return frames, SmartSegCache._ImageLoadType.Frame
        raise ValueError("State has no proper fields: image_ids, image_hashes or video_id")

    def _add_to_cache(
        self,
        names: Union[str, List[str]],
        images: Union[np.ndarray, List[np.ndarray]],
    ):
        if isinstance(names, str):
            names = [names]

        if isinstance(images, np.ndarray):
            images = [images]

        if len(images) != len(names):
            raise ValueError(
                f"Number of images and names do not match: {len(images)} != {len(names)}"
            )

        for fi, img in zip(names, images):
            self._cache[fi] = img

    def _image_name(self, id_or_hash: Any) -> str:
        return f"image_{id_or_hash}"

    def _frame_name(self, video_id: int, frame_index: int) -> str:
        return f"frame_{video_id}_{frame_index}"
