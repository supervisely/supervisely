import json
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from logging import Logger
from pathlib import Path
from threading import Lock, Thread
from typing import Any, BinaryIO, Callable, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
from cacheout import Cache as CacheOut
from cachetools import Cache, LRUCache, TTLCache
from fastapi import BackgroundTasks, FastAPI, Form, Request, UploadFile

import supervisely as sly
import supervisely.io.env as env
from supervisely._utils import batched
from supervisely.io.fs import list_files, silent_remove
from supervisely.video.video import VideoFrameReader


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
        self._base_dir = filepath

    def pop(self, *args, **kwargs):
        try:
            super().pop(*args, **kwargs)
        except Exception:
            sly.logger.warn("Cache data corrupted. Cleaning the cache...", exc_info=True)

            def _delitem(self, key):
                try:
                    size = self._Cache__size.pop(key)
                except:
                    size = 0
                self._Cache__data.pop(key, None)
                self._Cache__currsize -= size

            shutil.rmtree(self._base_dir, ignore_errors=True)
            for key in self.keys():
                try:
                    super().__delitem__(key, cache_delitem=_delitem)
                except:
                    pass

    def __delitem__(self, key: Any) -> None:
        self.__del_file(key)
        return super().__delitem__(key)

    def __del_file(self, key: str):
        cache_getitem = Cache.__getitem__
        filepath = cache_getitem(self, key)
        try:
            silent_remove(filepath)
        except TypeError:
            pass
        except:
            sly.logger.debug(f"File {filepath} was not deleted from cache", exc_info=True)
        else:
            sly.logger.debug(f"File {filepath} was deleted from cache")

    def __update_timer(self, key):
        try:
            # pylint: disable=no-member
            link = self._TTLCache__getlink(key)
            # pylint: disable=no-member
            if hasattr(link, "expire"):
                link.expire = self.timer() + self._TTLCache__ttl
            else:
                link.expires = self.timer() + self._TTLCache__ttl
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
        super().expire(time)
        deleted = set(existing_items.keys()).difference(self.__get_keys())
        if len(deleted) > 0:
            sly.logger.debug("Deleting expired items")
            for key in deleted:
                try:
                    silent_remove(existing_items[key])
                except TypeError:
                    pass
            sly.logger.debug(f"Deleted keys: {deleted}")
            sly.logger.trace(f"Existing files: {list_files(str(self._base_dir))}")

    def clear(self, rm_base_folder=True) -> None:
        while self.currsize > 0:
            self.popitem()
        if rm_base_folder:
            shutil.rmtree(self._base_dir)

    def save_image(self, key, image: Union[np.ndarray, BinaryIO, bytes], ext=".png") -> None:
        if not self._base_dir.exists():
            self._base_dir.mkdir()

        if ext is None or ext == "":
            ext = ".png"

        filepath = self._base_dir / Path(key).with_suffix(ext)
        self[key] = filepath

        if filepath.exists():
            sly.logger.debug(f"Rewrite image {str(filepath)}")
        if isinstance(image, np.ndarray):
            sly.image.write(str(filepath), image)
        elif isinstance(image, bytes):
            with open(filepath, "wb") as f:
                f.write(image)
        else:
            with open(filepath, "wb") as f:
                shutil.copyfileobj(image, f)

    def get_image_path(self, key: Any) -> Path:
        return self[key]

    def get_image(self, key: Any):
        return sly.image.read(str(self[key]))

    def save_video(self, key: Any, source: Union[str, Path, BinaryIO]) -> None:
        ext = ""
        if isinstance(source, Path):
            ext = source.suffix
        elif isinstance(source, str):
            ext = Path(source).suffix
        video_path = self._base_dir / f"video_{key}{ext}"
        self[key] = video_path

        if isinstance(source, (str, Path)):
            if str(source) != str(video_path):
                shutil.move(source, str(video_path))
        else:
            with open(video_path, "wb") as f:
                shutil.copyfileobj(source, f)
        sly.logger.debug(f"Video #{key} saved to {video_path}", extra={"video_id": key})

    def get_video_path(self, key: Any) -> Path:
        return self[key]

    def save_project_meta(self, key, value):
        self[key] = value

    def get_project_meta(self, project_meta_name):
        return self[project_meta_name]

    def copy_to(self, name, path):
        shutil.copyfile(str(self[name]), path)


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
        base_folder: str = env.smart_cache_container_dir(),
        log_progress: bool = False,
    ) -> None:
        self.is_persistent = is_persistent
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = Lock()
        self._load_queue = CacheOut(ttl=10 * 60)
        self.log_progress = log_progress
        self._download_executor = ThreadPoolExecutor(max_workers=5)

        if is_persistent:
            self._data_dir = Path(base_folder)
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._cache = PersistentImageTTLCache(maxsize, ttl, self._data_dir)
        else:
            self._cache = TTLCache(maxsize, ttl)

    def clear_cache(self):
        with self._lock:
            self._cache.clear(False)

    def download_image(self, api: sly.Api, image_id: int, related: bool = False):
        api.logger.debug(f"Download image #{image_id} to cache started")
        name = self._image_name(image_id)
        self._wait_if_in_queue(name, api.logger)

        if name not in self._cache:
            api.logger.debug(f"Adding image #{image_id} to cache")
            self._load_queue.set(name, image_id)
            if not related:
                img = api.image.download_np(image_id)
            else:
                img = api.pointcloud.download_related_image(image_id)
            self._add_to_cache(name, img)
            api.logger.debug(f"Added image #{image_id} to cache")
            return img
        else:
            api.logger.debug(f"Image #{image_id} found in cache")

        api.logger.debug(f"Get image #{image_id} from cache")
        return self._cache.get_image(name)

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
        image_key = self._image_name(img_hash)
        self._wait_if_in_queue(image_key, api.logger)

        if image_key not in self._cache:
            self._load_queue.set(image_key, img_hash)
            image = api.image.download_nps_by_hashes([img_hash])
            self._add_to_cache(image_key, image)
            return image
        return self._cache.get_image(image_key)

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

    def _read_frames_from_cached_video_iter(self, video_id, frame_indexes):
        video_path = self._cache.get_video_path(video_id)
        with VideoFrameReader(video_path, frame_indexes) as reader:
            for frame in reader.iterate_frames():
                yield frame

    def _read_frames_from_cached_video(
        self, video_id: int, frame_indexes: List[int]
    ) -> List[np.ndarray]:
        return [
            frame for frame in self._read_frames_from_cached_video_iter(video_id, frame_indexes)
        ]

    def get_frame_from_cache(self, video_id: int, frame_index: int) -> np.ndarray:
        name = self._frame_name(video_id, frame_index)
        if isinstance(self._cache, PersistentImageTTLCache):
            if name in self._cache:
                return self._cache.get_image(name)
            return self._read_frames_from_cached_video(video_id, [frame_index])[0]
        frame = self._cache.get(name)
        if frame is None:
            raise KeyError(f"Frame {frame_index} not found in video {video_id}")

    def get_video_frames_count(self, key):
        """
        Returns number of frames in the video
        """
        if not isinstance(self._cache, PersistentImageTTLCache):
            raise ValueError("Video frames count can be obtained only for persistent cache")
        video_path = self._cache.get_video_path(key)
        return VideoFrameReader(video_path).frames_count()

    def get_video_frame_size(self, key):
        """
        Returns height and width of the video frame. (h, w)
        """
        if not isinstance(self._cache, PersistentImageTTLCache):
            raise ValueError("Video frame size can be obtained only for persistent cache")
        video_path = self._cache.get_video_path(key)
        return VideoFrameReader(video_path).frame_size()

    def get_video_fps(self, key):
        """
        Returns fps of the video
        """
        if not isinstance(self._cache, PersistentImageTTLCache):
            raise ValueError("Video fps can be obtained only for persistent cache")
        video_path = self._cache.get_video_path(key)
        return VideoFrameReader(video_path).fps()

    def get_frames_from_cache(self, video_id: int, frame_indexes: List[int]) -> List[np.ndarray]:
        if isinstance(self._cache, PersistentImageTTLCache) and video_id in self._cache:
            return self._read_frames_from_cached_video(video_id, frame_indexes)
        else:
            return [
                self.get_frame_from_cache(video_id, frame_index) for frame_index in frame_indexes
            ]

    def frames_loader(
        self, api: sly.Api, video_id: int, frame_indexes: List[int]
    ) -> Generator[np.ndarray, None, None]:
        if not isinstance(self._cache, PersistentImageTTLCache):
            for frame_index in frame_indexes:
                yield self.download_frame(api, video_id, frame_index)
            return
        self.run_cache_task_manually(api, None, video_id=video_id)
        for i, frame_index in enumerate(frame_indexes):
            if video_id in self._cache:
                break
            yield self.download_frame(api, video_id, frame_index)
        if i < len(frame_indexes):
            for frame in self._read_frames_from_cached_video_iter(video_id, frame_indexes[i:]):
                yield frame

    def download_frame(self, api: sly.Api, video_id: int, frame_index: int) -> np.ndarray:
        name = self._frame_name(video_id, frame_index)
        self._wait_if_in_queue(name, api.logger)

        if name not in self._cache:
            if video_id in self._cache:
                api.logger.debug(
                    f"Get frame #{frame_index} for video #{video_id} from cache (video file)"
                )
                try:
                    return self.get_frame_from_cache(video_id, frame_index)
                except:
                    sly.logger.warning(
                        f"Frame {frame_index} not found in video {video_id}", exc_info=True
                    )

            self._load_queue.set(name, (video_id, frame_index))
            frame = api.video.frame.download_np(video_id, frame_index)
            self._add_to_cache(name, frame)
            api.logger.debug(f"Add frame #{frame_index} for video #{video_id} to cache")
            return frame

        api.logger.debug(f"Get frame #{frame_index} for video #{video_id} from cache")
        return self._cache.get_image(name)

    def download_frames(
        self, api: sly.Api, video_id: int, frame_indexes: List[int], **kwargs
    ) -> List[np.ndarray]:
        return_images = kwargs.get("return_images", True)
        redownload_video = kwargs.get("redownload_video", False)
        progress_cb = kwargs.get("progress_cb", None)

        if video_id in self._cache:
            try:
                frames = self.get_frames_from_cache(video_id, frame_indexes)
                if progress_cb is not None:
                    progress_cb(len(frame_indexes))
                if return_images:
                    return frames
            except:
                sly.logger.warning(
                    f"Frames {frame_indexes} not found in video {video_id}", exc_info=True
                )
                self._download_executor.submit(
                    self.download_video,
                    api,
                    video_id,
                    **{**kwargs, "return_images": False},
                )
        elif redownload_video:
            self._download_executor.submit(
                self.download_video,
                api,
                video_id,
                **{**kwargs, "return_images": False},
            )

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
            progress_cb=progress_cb,
            video_id=video_id,
        )

    def add_video_to_cache_by_io(self, video_id: int, video_io: BinaryIO) -> None:
        if isinstance(self._cache, PersistentImageTTLCache):
            with self._lock:
                self._cache.save_video(video_id, source=video_io)

    def add_video_to_cache(self, video_id: int, source: Union[str, Path, BinaryIO]) -> None:
        """
        Adds video to cache.
        """
        if isinstance(self._cache, PersistentImageTTLCache):
            with self._lock:
                self._cache.save_video(video_id, source)
                self._load_queue.delete(video_id)
            sly.logger.debug(f"Video #{video_id} added to cache", extra={"video_id": video_id})
        else:
            tmp_source = None
            if not isinstance(source, (str, Path)):
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    shutil.copyfileobj(source, f)
                    tmp_source = f.name
                    source = tmp_source
            try:
                with VideoFrameReader(source) as reader:
                    for frame_index, frame in enumerate(reader):
                        self.add_frame_to_cache(frame, video_id, frame_index)
            finally:
                if tmp_source is not None:
                    silent_remove(tmp_source)

    def add_image_to_cache(
        self, key: str, image: Union[np.ndarray, BinaryIO, bytes], ext=None
    ) -> np.ndarray:
        """
        Adds image to cache.
        """
        with self._lock:
            self._cache.save_image(key, image, ext)
            self._load_queue.delete(key)
            sly.logger.debug(f"Image {key} added to cache", extra={"image_id": key})
            return self._cache.get_image(key)

    def get_image_path(self, key) -> str:
        return str(self._cache.get_image_path(key))

    def get_video_path(self, key) -> str:
        return str(self._cache.get_video_path(key))

    def download_video(self, api: sly.Api, video_id: int, **kwargs) -> None:
        """
        Download video if needed and add it to cache. If video is already in cache, do nothing.
        If "return_images" in kwargs and is True, returns list of frames.
        """
        return_images = kwargs.get("return_images", True)
        progress_cb = kwargs.get("progress_cb", None)
        video_info = kwargs.get("video_info", None)
        if video_info is None:
            video_info = api.video.get_info_by_id(video_id)

        self._wait_if_in_queue(video_id, api.logger)
        if not video_id in self._cache:
            download_time = time.monotonic()
            self._load_queue.set(video_id, video_id)
            try:
                sly.logger.debug("Downloading video #%s", video_id)
                if progress_cb is None and self.log_progress:
                    size = video_info.file_meta.get("size", None)
                    if size is None:
                        size = "unknown"
                    else:
                        size = int(size)

                    prog_n = 0
                    prog_t = time.monotonic()

                    def _progress_cb(n):
                        nonlocal prog_n
                        nonlocal prog_t
                        prog_n += n
                        cur_t = time.monotonic()
                        if cur_t - prog_t > 3 or (isinstance(size, int) and prog_n >= size):
                            prog_t = cur_t
                            percent_str = ""
                            if isinstance(size, int):
                                percent_str = f" ({(prog_n*100) // size}%)"
                            prog_str = (
                                f"{(prog_n / 1000000):.2f}/{(size / 1000000):.2f} MB{percent_str}"
                            )
                            sly.logger.debug(
                                "Downloading video #%s: %s",
                                video_id,
                                prog_str,
                            )

                    progress_cb = _progress_cb
                temp_video_path = Path("/tmp/smart_cache").joinpath(
                    f"_{sly.rand_str(6)}_" + video_info.name
                )
                api.video.download_path(video_id, temp_video_path, progress_cb=progress_cb)
                self.add_video_to_cache(video_id, temp_video_path)
                download_time = time.monotonic() - download_time
                api.logger.debug(
                    f"Video #{video_id} downloaded to cache in {download_time:.2f} sec",
                    extra={"video_id": video_id, "download_time": download_time},
                )
                silent_remove(temp_video_path)
            except Exception as e:
                self._load_queue.delete(video_id)
                raise e
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
            self._wait_if_in_queue(video_id, sly.logger)
            self._load_queue.set(video_id, video_id)
            self.add_video_to_cache(video_id, files[0].file)

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

        self._download_executor.submit(self.cache_task, api=api, state=state)

    def set_project_meta(self, project_id, project_meta):
        pr_meta_name = self._project_meta_name(project_id)
        if isinstance(self._cache, PersistentImageTTLCache):
            self._cache.save_project_meta(pr_meta_name, project_meta)
        else:
            self._cache[pr_meta_name] = project_meta

    def get_project_meta(self, api: sly.Api, project_id: int):
        pr_meta_name = self._project_meta_name(project_id)
        if isinstance(self._cache, PersistentImageTTLCache):
            if pr_meta_name in self._cache:
                return self._cache.get_project_meta(pr_meta_name)
            project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
            self._cache.save_project_meta(pr_meta_name, project_meta)
            return project_meta
        else:
            if pr_meta_name in self._cache:
                return self._cache[pr_meta_name]
            project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
            self._cache[pr_meta_name] = project_meta
            return project_meta

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
            with self._lock:
                self._cache.save_image(name, img)
                self._load_queue.delete(name)

    def _image_name(self, id_or_hash: Union[str, int]) -> str:
        if isinstance(id_or_hash, int):
            return f"image_{id_or_hash}"
        hash_wo_slash = id_or_hash.replace("/", "-")
        return f"image_{hash_wo_slash}"

    def _frame_name(self, video_id: int, frame_index: int) -> str:
        return f"frame_{video_id}_{frame_index}"

    def _video_name(self, video_id: int, video_name: str) -> str:
        ext = Path(video_name).suffix
        name = f"video_{video_id}{ext}"
        return name

    def _project_meta_name(self, project_id: int) -> str:
        return f"project_meta_{project_id}"

    def _download_many(
        self,
        indexes: List[Union[int, str]],
        name_constructor: Callable[[int], str],
        load_generator: Callable[
            [List[int]],
            Generator[Tuple[Union[int, str], np.ndarray], None, None],
        ],
        logger: Logger,
        return_images: bool = True,
        progress_cb=None,
        video_id=None,
    ) -> Optional[List[np.ndarray]]:
        pos_by_name = {}
        all_frames = [None for _ in range(len(indexes))]

        def get_one_image(item):
            pos, hash_or_id = item
            if video_id in self._cache:
                try:
                    frame = self.get_frame_from_cache(video_id, hash_or_id)
                except Exception as e:
                    logger.error(
                        f"Error retrieving frame from cache: {repr(e)}. Frame will be re-downloaded",
                        exc_info=True,
                    )
                    ids_to_load.append(hash_or_id)
                    return pos, None
                return pos, frame
            try:
                image = self._cache.get_image(name_constructor(hash_or_id))
            except Exception as e:
                logger.error(
                    f"Error retrieving image from cache: {repr(e)}. Image will be re-downloaded",
                    exc_info=True,
                )
                ids_to_load.append(hash_or_id)
                return pos, None
            return pos, image

        position = 0
        batch_size = 4
        for batch in batched(indexes, batch_size):
            ids_to_load = []
            items = []
            for hash_or_id in batch:
                name = name_constructor(hash_or_id)
                self._wait_if_in_queue(name, logger)
                pos_by_name[name] = position
                if name not in self._cache and video_id not in self._cache:
                    self._load_queue.set(name, hash_or_id)
                    ids_to_load.append(hash_or_id)

                elif return_images is True:
                    items.append((position, hash_or_id))
                position += 1

            if len(items) > 0:
                with ThreadPoolExecutor(min(64, len(items))) as executor:
                    for pos, image in executor.map(get_one_image, items):
                        if image is None:
                            continue
                        all_frames[pos] = image
                        if progress_cb is not None:
                            progress_cb()

            download_time = time.monotonic()
            if len(ids_to_load) > 0:
                for id_or_hash, image in load_generator(ids_to_load):
                    name = name_constructor(id_or_hash)
                    self._add_to_cache(name, image)

                    if return_images:
                        pos = pos_by_name[name]
                        all_frames[pos] = image
                        if progress_cb is not None:
                            progress_cb()
            download_time = time.monotonic() - download_time

            # logger.debug(f"All stored files: {sorted(os.listdir(self.tmp_path))}")
            if ids_to_load:
                ids_to_load = list(ids_to_load)
                logger.debug(
                    f"Images/Frames added to cache: {ids_to_load} in {download_time:.2f} sec",
                    extra={"indexes": ids_to_load, "download_time": download_time},
                )
            found = set(batch).difference(ids_to_load)
            if found:
                logger.debug(f"Images/Frames found in cache: {list(found)}")

        if return_images:
            return all_frames
        return

    def _wait_if_in_queue(self, name, logger: Logger):
        if name in self._load_queue:
            logger.debug(f"Waiting for other task to load {name}")

        while name in self._load_queue:
            # TODO: time.sleep if slowdown
            time.sleep(0.1)
            continue

    def download_frames_to_paths(self, api, video_id, frame_indexes, paths, progress_cb=None):
        def _download_frame(frame_index):
            self.download_frame(api, video_id, frame_index)
            name = self._frame_name(video_id, frame_index)
            return frame_index, name

        def _download_and_save(this_frame_indexes, this_paths):
            if video_id in self._cache:
                for path, frame in zip(
                    this_paths,
                    self._read_frames_from_cached_video_iter(video_id, this_frame_indexes),
                ):
                    sly.image.write(path, frame)
                    if progress_cb is not None:
                        progress_cb()
                return

            futures = []
            frame_index_to_path = {}
            for frame_index, path in zip(this_frame_indexes[:5], this_paths[:5]):
                frame_index_to_path[frame_index] = path
                futures.append(self._download_executor.submit(_download_frame, frame_index))
            for future in as_completed(futures):
                frame_index, name = future.result()
                path = frame_index_to_path[frame_index]
                self._cache.copy_to(name, path)
                if progress_cb is not None:
                    progress_cb()
            if len(this_frame_indexes) > 5:
                _download_and_save(this_frame_indexes[5:], this_paths[5:])

        # optimization for frame read from video file
        frame_indexes, paths = zip(*sorted(zip(frame_indexes, paths), key=lambda x: x[0]))
        _download_and_save(frame_indexes, paths)
