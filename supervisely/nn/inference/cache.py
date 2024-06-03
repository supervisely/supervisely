import json
import os
import shutil
from enum import Enum
from logging import Logger
from pathlib import Path
from threading import Lock, Thread
from time import sleep
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
    def __init__(self, maxsize: int, ttl: int, filepath: Path):
        super().__init__(maxsize, ttl)
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
        self.__del_file(key)
        return super().__delitem__(key)

    def __del_file(self, key: str):
        cache_getitme = Cache.__getitem__
        filepath = cache_getitme(self, key)
        silent_remove(filepath)

    def __get_keys(self):
        # pylint: disable=no-member
        return self._TTLCache__links.keys()

    def expire(self, time=None):
        """Remove expired items from the cache."""
        existing = set(self.__get_keys())
        if time is None:
            # pylint: disable=no-member
            time = self._TTLCache__timer()
        # pylint: disable=no-member
        root = self._TTLCache__root
        curr = root.next
        # pylint: disable=no-member
        links = self._TTLCache__links
        cache_delitem = Cache.__delitem__
        while curr is not root and curr.expire < time:
            self.__del_file(curr.key)
            cache_delitem(self, curr.key)
            del links[curr.key]
            next = curr.next
            curr.unlink()
            curr = next
        deleted = existing.difference(self.__get_keys())
        if len(deleted) > 0:
            sly.logger.debug(f"Deleted keys: {deleted}")

    def clear(self, rm_base_folder=True) -> None:
        while self.currsize > 0:
            self.popitem()
        if rm_base_folder:
            shutil.rmtree(self._base_dir)

    def save_video(self, video_id: int, src_video_path: str) -> None:
        video_path = self._base_dir / f"video_{video_id}.{src_video_path.split('.')[-1]}"
        if src_video_path != str(video_path):
            shutil.move(src_video_path, str(video_path))
        super().__setitem__(video_id, video_path)

    def get_video_path(self, video_id: int) -> Path:
        return super().__getitem__(video_id)

    def save_project_meta(self, key, value):
        super().__setitem__(key, value)

    def get_project_meta(self, project_meta_name):
        return super().__getitem__(project_meta_name)


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
        self._is_persistent = is_persistent
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = Lock()
        self._load_queue = CacheOut(10 * 60)

        if is_persistent:
            self._data_dir = Path(base_folder)
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._cache = PersistentImageTTLCache(maxsize, ttl, self._data_dir)
        else:
            self._cache = TTLCache(maxsize, ttl)

    def clear_cache(self):
        with self._lock:
            self._cache.clear(False)

    def download_image(self, api: sly.Api, image_id: int):
        name = self._image_name(image_id)
        self._wait_if_in_queue(name, api.logger)

        if name not in self._cache:
            self._load_queue.set(name, image_id)
            api.logger.debug(f"Add image #{image_id} to cache")
            img = api.image.download_np(image_id)
            self._add_to_cache(name, img)
            return img

        api.logger.debug(f"Get image #{image_id} from cache")
        return self._cache[name]

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
        return self._cache[image_key]

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
        video_path = self._cache.get_video_path(video_id)
        if video_path is None or not video_path.exists():
            raise KeyError(f"Video {video_id} not found in cache")
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        prev_idx = -1
        for frame_index in frame_indexes:
            if frame_index != prev_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                raise KeyError(f"Frame {frame_index} not found in video {video_id}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            prev_idx = frame_index
        cap.release()
        return frames

    def get_frame_from_cache(self, video_id: int, frame_index: int) -> np.ndarray:
        name = self._frame_name(video_id, frame_index)
        if isinstance(self._cache, PersistentImageTTLCache):
            if name in self._cache:
                return self._cache.get(name)
            return self._read_frames_from_cached_video(video_id, [frame_index])[0]
        frame = self._cache.get(name)
        if frame is None:
            raise KeyError(f"Frame {frame_index} not found in video {video_id}")

    def get_frames_from_cache(self, video_id: int, frame_indexes: List[int]) -> List[np.ndarray]:
        if isinstance(self._cache, PersistentImageTTLCache) and video_id in self._cache:
            return self._read_frames_from_cached_video(video_id, frame_indexes)
        else:
            return [
                self.get_frame_from_cache(video_id, frame_index) for frame_index in frame_indexes
            ]

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
        return self._cache[name]

    def download_frames(
        self, api: sly.Api, video_id: int, frame_indexes: List[int], **kwargs
    ) -> List[np.ndarray]:
        return_images = kwargs.get("return_images", True)

        if video_id in self._cache:
            try:
                return self.get_frames_from_cache(video_id, frame_indexes)
            except:
                sly.logger.warning(
                    f"Frames {frame_indexes} not found in video {video_id}", exc_info=True
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
        )

    def add_video_to_cache(self, video_id: int, video_path: Path) -> None:
        """
        Adds video to cache.
        """
        if isinstance(self._cache, PersistentImageTTLCache):
            with self._lock:
                self._cache.save_video(video_id, str(video_path))
                self._load_queue.delete(video_id)
        else:
            cap = cv2.VideoCapture(str(video_path))
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
            cap.release()

    def download_video(self, api: sly.Api, video_id: int, **kwargs) -> None:
        """
        Download video if needed and add it to cache. If video is already in cache, do nothing.
        If "return_images" in kwargs and is True, returns list of frames.
        """
        return_images = kwargs.get("return_images", True)
        progress_cb = kwargs.get("progress_cb", None)

        self._wait_if_in_queue(video_id, api.logger)
        if not video_id in self._cache:
            self._load_queue.set(video_id, video_id)
            sly.logger.debug("Downloading video #%s", video_id)
            video_info = api.video.get_info_by_id(video_id)
            temp_video_path = Path("/tmp/smart_cache").joinpath(
                f"_{sly.rand_str(6)}_" + video_info.name
            )
            api.video.download_path(video_id, temp_video_path, progress_cb=progress_cb)
            self.add_video_to_cache(video_id, temp_video_path)
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
            self._wait_if_in_queue(video_id, sly.logger)
            self._load_queue.set(video_id, video_id)
            self.add_video_to_cache(video_id, str(temp_video_path))

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
        if self._is_persistent:
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
                self._cache[name] = img
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
        name_cunstructor: Callable[[int], str],
        load_generator: Callable[
            [List[int]],
            Generator[Tuple[Union[int, str], np.ndarray], None, None],
        ],
        logger: Logger,
        return_images: bool = True,
    ) -> Optional[List[np.ndarray]]:
        indexes_to_load = []
        pos_by_name = {}
        all_frames = [None for _ in range(len(indexes))]

        for pos, hash_or_id in enumerate(indexes):
            name = name_cunstructor(hash_or_id)
            self._wait_if_in_queue(name, logger)

            if name not in self._cache:
                self._load_queue.set(name, hash_or_id)
                indexes_to_load.append(hash_or_id)
                pos_by_name[name] = pos
            elif return_images is True:
                all_frames[pos] = self._cache[name]

        if len(indexes_to_load) > 0:
            for id_or_hash, image in load_generator(indexes_to_load):
                name = name_cunstructor(id_or_hash)
                self._add_to_cache(name, image)

                if return_images:
                    pos = pos_by_name[name]
                    all_frames[pos] = image

        logger.debug(f"All stored files: {sorted(os.listdir(self.tmp_path))}")
        logger.debug(f"Images/Frames added to cache: {indexes_to_load}")
        logger.debug(f"Images/Frames found in cache: {set(indexes).difference(indexes_to_load)}")

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
