import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from enum import Enum
from logging import Logger
from pathlib import Path
from threading import Lock, RLock, Thread
from time import monotonic, sleep
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
from cacheout import Cache as CacheOut
from cachetools import Cache, LRUCache, TTLCache
from fastapi import BackgroundTasks, FastAPI, Form, Request, UploadFile

import supervisely as sly
from supervisely.io.fs import silent_remove


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
    TIMESTEP = 0.0001

    def __init__(self, maxsize: int, ttl: int, filepath: Path):
        super().__init__(maxsize, ttl)
        self._base_dir = filepath
        self._lock = RLock()
        self._lockmap = {}

    @contextmanager
    def acquire_lock(self, key, timeout=None):
        retry = True
        start = monotonic()
        try:
            while retry:
                retry = False
                with self._lock:
                    if not self._lockmap.get(key, False):
                        self._lockmap[key] = True
                    else:
                        retry = True
                        sleep(self.TIMESTEP)
                        if timeout is not None:
                            if monotonic() - start > timeout:
                                raise TimeoutError()
            yield
        finally:
            with self._lock:
                try:
                    del self._lockmap[key]
                except KeyError:
                    pass

    def __clear_cache(self):
        sly.logger.warn("Clearing the cache due to an error...")
        shutil.rmtree(self._base_dir, ignore_errors=True)
        self.__init__(self.maxsize, self.ttl, self._base_dir)

    def pop(self, *args, **kwargs):
        try:
            super().pop(*args, **kwargs)
        except Exception:
            sly.logger.warn("Cache data corrupted", exc_info=True)
            self.__clear_cache()

    def __delitem__(self, key: Any) -> None:
        with self.acquire_lock(key):
            self.__del_file(key)
            super().__delitem__(key)

    def __del_file(self, key: str):
        cache_getitem = Cache.__getitem__
        filepath = cache_getitem(self, key)
        try:
            silent_remove(filepath)
        except TypeError:
            pass

    def __update_timer(self, key):
        try:
            # pylint: disable=no-member
            link = self._TTLCache__getlink(key)
            # pylint: disable=no-member
            link.expire = self.timer() + self.ttl
        except KeyError:
            return

    def __getitem__(self, key: Any) -> Any:
        self.__update_timer(key)
        return super().__getitem__(key)

    def __get_keys(self):
        # pylint: disable=no-member
        return self._TTLCache__links.keys()

    def expire(self, time=None):
        """Remove expired items from the cache."""
        # pylint: disable=no-member
        existing_items = self._Cache__data.copy()
        try:
            super().expire(time)
        except:
            sly.logger.warn("Cache data corrupted", exc_info=True)
            self.__clear_cache()
        deleted = set(existing_items.keys()).difference(self.__get_keys())
        if len(deleted) > 0:
            for key in deleted:
                try:
                    silent_remove(existing_items[key])
                except TypeError:
                    pass
            sly.logger.debug(f"Deleted keys: {deleted}")

    def clear(self, rm_base_folder=True) -> None:
        while self.currsize > 0:
            self.popitem()
        if rm_base_folder:
            shutil.rmtree(self._base_dir)

    def save_image(self, key, image: np.ndarray) -> None:
        with self.acquire_lock(key):
            if not self._base_dir.exists():
                self._base_dir.mkdir()

            filepath = self._base_dir / f"{str(key)}.png"
            self[key] = filepath

            if filepath.exists():
                sly.logger.debug(f"Rewrite image {str(filepath)}")
            sly.image.write(str(filepath), image)

    def get_image_path(self, key: Any) -> Path:
        return self[key]

    def get_image(self, key: Any):
        with self.acquire_lock(key):
            return sly.image.read(str(self[key]))

    def save_video(self, video_id: int, src_video_path: str) -> None:
        with self.acquire_lock(video_id):
            video_path = self._base_dir / f"video_{video_id}.{src_video_path.split('.')[-1]}"
            self[video_id] = video_path
            if src_video_path != str(video_path):
                shutil.move(src_video_path, str(video_path))
            sly.logger.debug(f"Saved video to {video_path}")

    @contextmanager
    def open_video(self, video_id: int):
        with self.acquire_lock(video_id):
            video_path = self.get_video_path(video_id)
            if video_path is None or not video_path.exists():
                raise KeyError(f"Video {video_id} not found in cache")
            cap = cv2.VideoCapture(str(video_path))
            try:
                yield cap
            finally:
                cap.release()

    def get_video_path(self, video_id: int) -> Path:
        return self.get(video_id, None)

    def save_project_meta(self, key, value):
        with self.acquire_lock(key):
            self[key] = value

    def get_project_meta(self, project_meta_name):
        return self[project_meta_name]


class InferenceImageCache:
    class _LoadType(Enum):
        ImageId: str = "IMAGE"
        ImageHash: str = "HASH"
        Frame: str = "FRAME"
        Video: str = "VIDEO"

    def __init__(
        self,
        maxsize: int,
        ttl: int,
        is_persistent: bool = True,
        base_folder: str = sly.env.smart_cache_container_dir(),
    ) -> None:
        self.is_persistent = is_persistent
        self._maxsize = maxsize
        self._ttl = ttl
        self._load_queue = CacheOut(10 * 60)

        if is_persistent:
            self._data_dir = Path(base_folder)
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._cache = PersistentImageTTLCache(maxsize, ttl, self._data_dir)
        else:
            self._cache = TTLCache(maxsize, ttl)

    def clear_cache(self):
        self._cache.clear(False)

    def _get_image_from_cache(self, key: Any) -> np.ndarray:
        if isinstance(self._cache, PersistentImageTTLCache):
            return self._cache.get_image(key)
        return self._cache[key]

    def download_image(self, api: sly.Api, image_id: int):
        img = None
        name = self._image_name(image_id)
        try:
            self._wait_if_in_queue(name, api.logger)
            if name in self._cache:
                sly.logger.debug(f"Get image #{image_id} from cache")
                return self._get_image_from_cache(name)

            self._load_queue.set(name, image_id)
            sly.logger.debug(f"Add image #{image_id} to cache")
            img = api.image.download_np(image_id)
            self._add_to_cache(name, img)
        except:
            self._load_queue.delete(name)
            api.logger.warning(f"Error downloading image #{image_id} using cache", exc_info=True)
            if img is None:
                img = api.image.download_np(image_id)
        return img

    def download_images(self, api: sly.Api, dataset_id: int, image_ids: List[int], **kwargs):
        return_images = kwargs.get("return_images", True)

        def load_generator(image_ids: List[int]):
            return api.image.download_nps_generator(dataset_id, image_ids)

        return self._download_many(
            image_ids,
            self._image_name,
            load_generator,
            api.logger,
            return_images,
        )

    def download_image_by_hash(self, api: sly.Api, img_hash: str) -> np.ndarray:
        image = None
        image_key = self._image_name(img_hash)
        try:
            self._wait_if_in_queue(image_key, api.logger)
            if image_key in self._cache:
                api.logger.debug(f"Get image #{img_hash} from cache")
                return self._get_image_from_cache(image_key)

            self._load_queue.set(image_key, img_hash)
            sly.logger.debug(f"Add image #{img_hash} to cache")
            image = api.image.download_nps_by_hashes([img_hash])[0]
            self._add_to_cache(image_key, image)
        except Exception:
            self._load_queue.delete(image_key)
            api.logger.warning(
                f"Error while downloading image #{img_hash} using cache", exc_info=True
            )
            if image is None:
                image = api.image.download_nps_by_hashes([img_hash])[0]
        return image

    def download_images_by_hashes(
        self, api: sly.Api, img_hashes: List[str], **kwargs
    ) -> List[np.ndarray]:
        return_images = kwargs.get("return_images", True)

        def load_generator(img_hashes: List[str]):
            return api.image.download_nps_by_hashes_generator(img_hashes)

        return self._download_many(
            img_hashes,
            self._image_name,
            load_generator,
            api.logger,
            return_images,
        )

    def _read_frames_from_cached_video(
        self, video_id: int, frame_indexes: List[int]
    ) -> List[np.ndarray]:
        with self._cache.open_video(video_id) as cap:
            frames = []
            prev_idx = -1
            for frame_index in frame_indexes:
                if frame_index != prev_idx + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    raise KeyError(
                        f"Frame {frame_index} not found in cached video #{video_id} file"
                    )
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                prev_idx = frame_index
            return frames

    def _get_frame_from_cache(self, video_id: int, frame_index: int) -> np.ndarray:
        name = self._frame_name(video_id, frame_index)
        if name in self._cache:
            return self._get_image_from_cache(name)
        if video_id in self._cache:
            return self._read_frames_from_cached_video(video_id, [frame_index])[0]
        raise KeyError(f"Frame {frame_index} of video #{video_id} not found in cache")

    def get_frames_from_cache(self, video_id: int, frame_indexes: List[int]) -> List[np.ndarray]:
        if isinstance(self._cache, PersistentImageTTLCache) and video_id in self._cache:
            return self._read_frames_from_cached_video(video_id, frame_indexes)
        else:
            return [
                self._get_frame_from_cache(video_id, frame_index) for frame_index in frame_indexes
            ]

    def download_frame(self, api: sly.Api, video_id: int, frame_index: int) -> np.ndarray:
        frame = None
        try:
            frame = self._get_frame_from_cache(video_id, frame_index)
            sly.logger.debug(f"Get frame {frame_index} of video #{video_id} from cache")
            return frame
        except KeyError:
            pass
        name = self._frame_name(video_id, frame_index)
        try:
            self._wait_if_in_queue(name, api.logger)
            if name in self._cache:
                sly.logger.debug(f"Get frame {frame_index} of video #{video_id} from cache")
                return self._get_image_from_cache(name)
            self._load_queue.set(name, (video_id, frame_index))
            frame = api.video.frame.download_np(video_id, frame_index)
            self._add_to_cache(name, frame)
            api.logger.debug(f"Add frame {frame_index} of video #{video_id} to cache")
        except Exception:
            self._load_queue.delete(name)
            sly.logger.debug(
                f"Error downloading frame {frame_index} of video #{video_id} using cache",
                exc_info=True,
            )
            if frame is None:
                return api.video.frame.download_np(video_id, frame_index)
        return frame

    def download_frames(
        self, api: sly.Api, video_id: int, frame_indexes: List[int], **kwargs
    ) -> List[np.ndarray]:
        return_images = kwargs.get("return_images", True)
        redownload_video = kwargs.get("redownload_video", False)

        try:
            if video_id in self._cache:
                return self.get_frames_from_cache(video_id, frame_indexes)
        except Exception:
            sly.logger.debug(
                f"Error getting frames from cache",
                exc_info=True,
                extra={"video_id": video_id, "frame_indexes": frame_indexes},
            )
        if redownload_video:
            Thread(
                target=self.download_video,
                args=(api, video_id),
                kwargs={"return_images": False},
            ).start()

        def name_constuctor(frame_index: int):
            return self._frame_name(video_id, frame_index)

        def load_generator(frame_indexes: List[int]):
            return api.video.frame.download_nps_generator(video_id, frame_indexes)

        return self._download_many(
            frame_indexes,
            name_constuctor,
            load_generator,
            api.logger,
            return_images,
        )

    def _add_video_to_cache(self, video_id: int, video_path: Path) -> None:
        """
        Adds video to cache.
        """
        if isinstance(self._cache, PersistentImageTTLCache):
            self._cache.save_video(video_id, str(video_path))
            self._load_queue.delete(video_id)
        else:
            cap = cv2.VideoCapture(str(video_path))
            try:
                frame_index = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while cap.isOpened():
                    while self._frame_name(video_id, frame_index) in self._cache:
                        frame_index += 1
                    if frame_index >= total_frames:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = np.array(frame)
                    self.add_frame_to_cache(frame, video_id, frame_index)
                    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            finally:
                cap.release()

    def download_video(self, api: sly.Api, video_id: int, **kwargs) -> None:
        """
        Download video if needed and add it to cache. If video is already in cache, do nothing.
        If "return_images" in kwargs and is True, returns list of frames.
        """
        return_images = kwargs.get("return_images", True)
        progress_cb = kwargs.get("progress_cb", None)

        video_info = api.video.get_info_by_id(video_id)
        self._wait_if_in_queue(video_id, api.logger)
        if not video_id in self._cache:
            self._load_queue.set(video_id, video_id)
            sly.logger.debug("Downloading video #%s", video_id)
            temp_video_path = Path("/tmp/smart_cache").joinpath(
                f"_{sly.rand_str(6)}_" + video_info.name
            )
            api.video.download_path(video_id, temp_video_path, progress_cb=progress_cb)
            self._add_video_to_cache(video_id, temp_video_path)
        if return_images:
            return self.get_frames_from_cache(video_id, list(range(video_info.frames_count)))

    def add_cache_endpoint(self, server: FastAPI):
        @server.post("/smart_cache")
        def cache_endpoint(request: Request, task: BackgroundTasks):
            task.add_task(
                self.cache_task,
                api=request.state.api,
                state=request.state.state,
            )
            return {"message": "Cache task started."}

    def add_cache_files_endpoint(self, server: FastAPI):
        @server.post("/smart_cache_files")
        def cache_files_endpoint(
            request: Request,
            task: BackgroundTasks,
            files: List[UploadFile],
            settings: str = Form("{}"),
        ):
            state = json.loads(settings)
            task.add_task(
                self.cache_files_task,
                files=files,
                state=state,
            )
            return {"message": "Cache task started."}

    def cache_task(self, api: sly.Api, state: dict):
        if "server_address" in state and "api_token" in state:
            api = sly.Api(state["server_address"], state["api_token"])
        api.logger.debug("Request state in cache endpoint", extra=state)
        image_ids, task_type = self._parse_state(state)
        kwargs = {"return_images": False}

        if task_type is InferenceImageCache._LoadType.ImageId:
            if "dataset_id" in state:
                self.download_images(api, state["dataset_id"], image_ids, **kwargs)
            else:
                for img_id in image_ids:
                    self.download_image(api, img_id)
        elif task_type is InferenceImageCache._LoadType.ImageHash:
            self.download_images_by_hashes(api, image_ids, **kwargs)
        elif task_type is InferenceImageCache._LoadType.Frame:
            video_id = state["video_id"]
            self.download_frames(api, video_id, image_ids, **kwargs)
        elif task_type is InferenceImageCache._LoadType.Video:
            video_id = image_ids
            self.download_video(api, video_id, **kwargs)

    def add_frame_to_cache(self, frame: np.ndarray, video_id: int, frame_index: int):
        name = self._frame_name(video_id, frame_index)
        self._add_to_cache(name, frame)

    def cache_files_task(self, files: List[UploadFile], state: dict):
        sly.logger.debug("Request state in cache endpoint", extra=state)
        image_ids, task_type = self._parse_state(state)

        if task_type in (
            InferenceImageCache._LoadType.ImageId,
            InferenceImageCache._LoadType.ImageHash,
        ):
            for image_id, file in zip(image_ids, files):
                image = sly.image.read_bytes(file.file.read())
                name = self._image_name(image_id)
                self._add_to_cache(name, image)
        elif task_type is InferenceImageCache._LoadType.Frame:
            video_id = state["video_id"]
            for frame_index, file in zip(image_ids, files):
                frame = sly.image.read_bytes(file.file.read())
                self.add_frame_to_cache(frame, video_id, frame_index)
        elif task_type is InferenceImageCache._LoadType.Video:
            video_id = image_ids
            temp_video_path = Path("/tmp/smart_cache").joinpath(
                f"_{sly.rand_str(6)}_" + files[0].file.name
            )
            with open(temp_video_path, "wb") as f:
                shutil.copyfileobj(files[0].file, f)
            self._add_video_to_cache(video_id, str(temp_video_path))

    def run_cache_task_manually(
        self,
        api: sly.Api,
        list_of_ids_ranges_or_hashes: List[Union[str, int, List[int]]],
        *,
        dataset_id: Optional[int] = None,
        video_id: Optional[int] = None,
    ) -> None:
        """
        Run cache_task in new thread.

        :param api: supervisely api
        :type api: sly.Api
        :param list_of_ids_ranges_or_hashes: information abount images/frames need to be loaded;
        to download images, pass list of integer IDs (`dataset_id` requires)
        or list of hash strings (`dataset_id` could be None);
        to download frames, pass list of pairs of indices of the first and last frame
        and `video_id` (ex.: [[1, 3], [5, 5], [7, 10]]);
        to download video, pass None and `video_id` (only for persistent cache)
        :type list_of_ids_ranges_or_hashes: List[Union[str, int, List[int]]]
        :param dataset_id: id of dataset on supervisely platform; default is None
        :type dataset_id: Optional[int]
        :param video_id: id of video on supervisely platform; default is None
        :type video_id: Optional[int]
        """
        state = {}
        if list_of_ids_ranges_or_hashes is None:
            api.logger.debug("Got a task to add video to cache")
            if not isinstance(self._cache, PersistentImageTTLCache):
                raise ValueError("Video can be added only to persistent cache")
            state["video_id"] = video_id
        elif isinstance(list_of_ids_ranges_or_hashes[0], str):
            api.logger.debug("Got a task to add images using hash")
            state["image_hashes"] = list_of_ids_ranges_or_hashes
        elif video_id is None:
            if dataset_id is None:
                api.logger.error("dataset_id or video_id must be defined if not hashes are used")
                return
            api.logger.debug("Got a task to add images using IDs")
            state["image_ids"] = list_of_ids_ranges_or_hashes
            state["dataset_id"] = dataset_id
        else:
            api.logger.debug("Got a task to add frames")
            state["video_id"] = video_id
            state["frame_ranges"] = list_of_ids_ranges_or_hashes

        thread = Thread(target=self.cache_task, kwargs={"api": api, "state": state})
        thread.start()

    def _set_project_meta(self, project_id, project_meta):
        pr_meta_name = self._project_meta_name(project_id)
        if isinstance(self._cache, PersistentImageTTLCache):
            self._cache.save_project_meta(pr_meta_name, project_meta)
        else:
            self._cache[pr_meta_name] = project_meta

    def set_project_meta(self, project_id, project_meta):
        try:
            self._set_project_meta(project_id, project_meta)
        except Exception:
            sly.logger.debug(
                f"Failed to update meta for project #{project_id} in cache", exc_info=True
            )

    def _get_project_meta(self, project_id):
        pr_meta_name = self._project_meta_name(project_id)
        if isinstance(self._cache, PersistentImageTTLCache):
            return self._cache.get_project_meta(pr_meta_name)
        return self._cache[pr_meta_name]

    def download_project_meta(self, api: sly.Api, project_id: int):
        try:
            pr_meta_name = self._project_meta_name(project_id)
            if pr_meta_name in self._cache:
                return self._get_project_meta(project_id)
        except:
            sly.logger.debug(
                f"Error getting meta of project #{project_id} from cache", exc_info=True
            )
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        self.set_project_meta(project_id, project_meta)
        return project_meta

    def get_project_meta(self, api: sly.Api, project_id: int):
        """Deprecated. Use download_project_meta instead."""
        return self.download_project_meta(api, project_id)

    @property
    def ttl(self):
        return self._ttl

    @property
    def tmp_path(self):
        if self.is_persistent:
            return str(self._data_dir)
        return None

    def _parse_state(self, state: dict) -> Tuple[List[Any], _LoadType]:
        if "image_ids" in state:
            return state["image_ids"], InferenceImageCache._LoadType.ImageId
        elif "image_hashes" in state:
            return state["image_hashes"], InferenceImageCache._LoadType.ImageHash
        elif "video_id" in state:
            if "frame_ranges" in state:
                frame_ranges = state["frame_ranges"]
                frames = []
                for fr_range in frame_ranges:
                    shift = 1
                    if fr_range[0] > fr_range[1]:
                        shift = -1
                    start, end = fr_range[0], fr_range[1] + shift
                    frames.extend(list(range(start, end, shift)))
                return frames, InferenceImageCache._LoadType.Frame
            elif "frame_indexes" in state:
                return state["frame_indexes"], InferenceImageCache._LoadType.Frame
            return state["video_id"], InferenceImageCache._LoadType.Video
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

        for name, img in zip(names, images):
            try:
                self._cache.save_image(name, img)
            finally:
                self._load_queue.delete(name)

    def _image_name(self, id_or_hash: Union[str, int]) -> str:
        if isinstance(id_or_hash, int):
            return f"image_{id_or_hash}"
        hash_wo_slash = id_or_hash.replace("/", "-")
        return f"image_{hash_wo_slash}"

    def _frame_name(self, video_id: int, frame_index: int) -> str:
        return f"frame_{video_id}_{frame_index}"

    def _video_name(self, video_id: int, video_name: str) -> str:
        return f"video_{video_id}.{video_name.split('.')[-1]}"

    def _project_meta_name(self, project_id: int) -> str:
        return f"project_meta_{project_id}"

    def _download_many(
        self,
        indexes: List[Union[int, str]],
        name_cunstructor: Callable[[Union[int, str]], str],
        load_generator: Callable[
            [List[int]],
            Generator[Tuple[Union[int, str], np.ndarray], None, None],
        ],
        logger: Logger,
        return_images: bool = True,
    ) -> Optional[List[np.ndarray]]:
        ids_to_load = []
        added_to_cache = []
        pos_by_name = {}
        all_frames = [None for _ in range(len(indexes))]
        items = []

        for pos, hash_or_id in enumerate(indexes):
            name = name_cunstructor(hash_or_id)
            self._wait_if_in_queue(name, logger)
            if not name in self._cache:
                self._load_queue.set(name, hash_or_id)
                ids_to_load.append(hash_or_id)
                pos_by_name[name] = pos
            elif return_images is True:
                items.append((pos, name))

        def get_one_image(item):
            pos, name = item
            try:
                return pos, self._get_image_from_cache(name)
            except Exception:
                logger.debug(f"Error reading item #{hash_or_id} from cache", exc_info=True)
                self._load_queue.set(name, hash_or_id)
                ids_to_load.append(hash_or_id)
                return pos, None

        if len(items) > 0:
            with ThreadPoolExecutor(min(64, len(items))) as executor:
                for pos, image in executor.map(get_one_image, items):
                    all_frames[pos] = image

        if len(ids_to_load) > 0:
            for id_or_hash, image in load_generator(ids_to_load):
                name = name_cunstructor(id_or_hash)
                try:
                    self._add_to_cache(name, image)
                    added_to_cache.append(id_or_hash)
                except Exception:
                    logger.debug(f"Error adding item #{id_or_hash} to cache", exc_info=True)

                if return_images:
                    pos = pos_by_name[name]
                    all_frames[pos] = image

        logger.debug(f"Images/Frames loaded: {ids_to_load}")
        logger.debug(f"Images/Frames added to cache: {added_to_cache}")
        logger.debug(f"Images/Frames found in cache: {set(indexes).difference(ids_to_load)}")

        if return_images:
            return all_frames
        return

    def _wait_if_in_queue(self, name, logger: Logger):
        if name in self._load_queue:
            logger.debug(f"Waiting for other task to load {name}")

        while name in self._load_queue:
            # TODO: sleep if slowdown
            sleep(0.1)
            continue
