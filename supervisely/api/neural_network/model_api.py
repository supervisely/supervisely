import atexit
import inspect
import os
import tempfile
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import requests

from supervisely._utils import get_valid_kwargs, logger, rand_str
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagValueType
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.image_api import ImageInfo
from supervisely.api.module_api import ApiField
from supervisely.api.project_api import ProjectInfo
from supervisely.api.task_api import TaskApi
from supervisely.api.video.video_api import VideoInfo
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging._video import ALLOWED_VIDEO_EXTENSIONS
from supervisely.imaging.image import SUPPORTED_IMG_EXTS
from supervisely.imaging.image import read as read_image
from supervisely.imaging.image import read_bytes as read_image_bytes
from supervisely.imaging.image import write as write_image
from supervisely.io.fs import (
    clean_dir,
    dir_empty,
    dir_exists,
    ensure_base_path,
    file_exists,
    get_file_ext,
    list_files,
    mkdir,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.video.video import VideoFrameReader

if TYPE_CHECKING:
    from supervisely.api.api import Api


class PredictionDTO:
    _temp_dir = os.path.join(tempfile.gettempdir(), "prediction_files")
    __cleanup_registered = False

    def __init__(
        self,
        annotation_json: Dict,
        source: Union[str, int] = None,
        model_meta: Optional[Union[ProjectMeta, Dict]] = None,
        name: Optional[str] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
        project_id: Optional[int] = None,
        dataset_id: Optional[int] = None,
        image_id: Optional[int] = None,
        video_id: Optional[int] = None,
        frame_index: Optional[int] = None,
    ):
        self.source = source
        self.annotation_json = annotation_json
        self.model_meta = model_meta
        if isinstance(self.model_meta, dict):
            self.model_meta = ProjectMeta.from_json(self.model_meta)

        self.name = name
        self.path = path
        self.url = url
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.image_id = image_id
        self.video_id = video_id
        self.frame_index = frame_index

        self._annotation = None
        self._boxes = None
        self._masks = None
        self._classes = None
        self._scores = None

        mkdir(self._temp_dir)
        if not PredictionDTO.__cleanup_registered:
            atexit.register(self._clear_temp_files)
            PredictionDTO.__cleanup_registered = True

    def _init_geometries(self):

        def _get_confidence(label: Label):
            for tag_name in ["confidence", "conf", "score"]:
                conf_tag: Tag = label.tags.get(tag_name, None)
                if conf_tag is not None and conf_tag.meta.value_type != str(
                    TagValueType.ANY_NUMBER
                ):
                    conf_tag = None
                if conf_tag is not None:
                    return conf_tag.value
            return 1

        self._boxes = []
        self._masks = []
        self._classes = []
        self._scores = []
        for label in self.annotation.labels:
            self._classes.append(label.obj_class.name)
            self._scores.append(_get_confidence(label))
            if isinstance(label.geometry, Rectangle):
                rect = label.geometry
            else:
                rect = label.geometry.to_bbox()
            if isinstance(label.geometry, Bitmap):
                self._masks.append(label.geometry.get_mask(self.annotation.img_size))

            self._boxes.append(
                np.array(
                    [
                        rect.top,
                        rect.left,
                        rect.bottom,
                        rect.right,
                    ]
                )
            )
        self._boxes = np.array(self._boxes)
        self._masks = np.array(self._masks)

    @property
    def boxes(self):
        if self._boxes is None:
            self._init_geometries()
        return self._boxes

    @property
    def masks(self):
        if self._masks is None:
            self._init_geometries()
        return self._masks

    @property
    def classes(self):
        if self._classes is None:
            self._init_geometries()
        return self._classes

    @property
    def scores(self):
        if self._scores is None:
            self._init_geometries()
        return self._scores

    @property
    def annotation(self) -> Annotation:
        if self._annotation is None:
            if self.model_meta is None:
                raise ValueError("Model meta is not provided. Cannot create annotation.")
            self._annotation = Annotation.from_json(self.annotation_json, self.model_meta)
        return self._annotation

    @property
    def class_idxs(self) -> np.ndarray:
        if self.model_meta is None:
            raise ValueError("Model meta is not provided. Cannot create class indexes.")
        cls_name_to_idx = {
            obj_class.name: i for i, obj_class in enumerate(self.model_meta.obj_classes)
        }
        return np.array([cls_name_to_idx[class_name] for class_name in self.classes])

    @classmethod
    def from_json(cls, json_data: Dict, source=None, model_meta: Optional[ProjectMeta] = None):
        kwargs = get_valid_kwargs(
            json_data,
            PredictionDTO.__init__,
            exclude=["self", "annotation", "source", "model_meta"],
        )
        if "annotation" in json_data:
            annotation_json = json_data["annotation"]
        else:
            annotation_json = json_data
        return cls(
            annotation_json=annotation_json,
            source=source,
            model_meta=model_meta,
            **kwargs,
        )

    def to_json(self):
        return {
            "source": self.source,
            "annotation": self.annotation_json,
            "name": self.name,
            "path": self.path,
            "url": self.url,
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "image_id": self.image_id,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
        }

    def _clear_temp_files(self):
        if not dir_empty(self._temp_dir):
            clean_dir(self._temp_dir)

    def load_image(self, api: Optional["Api"] = None) -> np.ndarray:
        if self.frame_index is None:
            if self.path is not None:
                return read_image(self.path)
            if self.url is not None:
                ext = get_file_ext(self.url)
                if ext == "":
                    ext = ".jpg"
                r = requests.get(self.url, stream=True, timeout=60)
                r.raise_for_status()
                return read_image_bytes(r.content)
            if self.image_id is not None:
                try:
                    from supervisely.api.api import Api

                    if api is None:
                        api = Api()
                    return api.image.download_np(self.image_id)
                except Exception as e:
                    raise RuntimeError("Failed to load image by ID") from e
            raise ValueError("Cannot load image. No path, URL, image ID, or frame_index provided.")
        if self.video_id is not None:
            if self.frame_index is None:
                raise ValueError("Frame index is not provided for video.")
            try:
                from supervisely.api.api import Api

                if api is None:
                    api = Api()
                return api.video.frame.download_np(self.video_id, self.frame_index)
            except Exception as e:
                raise RuntimeError("Failed to load frame by video ID") from e
        if self.path is not None:
            return next(VideoFrameReader(self.path, frame_indexes=[self.frame_index]))
        if self.url is not None:
            video_name = Path(self.url).name
            video_path = Path(self._temp_dir) / video_name
            mkdir(self._temp_dir)
            if not video_path.exists():
                with requests.get(self.url, stream=True, timeout=10 * 60) as r:
                    r.raise_for_status()
                    with open(video_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            return next(VideoFrameReader(video_path, frame_indexes=[self.frame_index]))
        raise ValueError("Cannot load frame. No path, URL or video ID provided.")

    def visualize(
        self,
        save_path: Optional[str] = None,
        color: Optional[Tuple[int, int, int]] = None,
        thickness: Optional[int] = 1,
        draw_tags: Optional[bool] = False,
        fill_rectangles: Optional[bool] = True,
        draw_class_names: Optional[bool] = False,
        api: Optional["Api"] = None,
    ) -> np.ndarray:
        img = self.load_image(api)
        self.annotation.draw(img, color, thickness, draw_tags, fill_rectangles, draw_class_names)
        if save_path is None:
            return img
        if Path(save_path).suffix == "":
            # is a directory
            if self.name is not None:
                name = self.name
                if self.frame_index is not None:
                    name = f"{name}_{self.frame_index}"
            elif self.image_id is not None:
                name = str(self.image_id)
            elif self.video_id is not None:
                name = str(self.video_id)
            elif self.path is not None:
                name = Path(self.path).name
                if self.frame_index is not None:
                    name = f"{name}_{self.frame_index}"
            else:
                name = f"{rand_str(6)}"
            name = f"vis_{name}.png"
            save_path = os.path.join(save_path, name)
        ensure_base_path(save_path)
        write_image(save_path, img)
        logger.info("Visualization for prediction saved to %s", save_path)
        return img


class InferenceSession:

    class _Iterator:
        def __init__(self, source: Any, iterator: Iterator, **kwargs):
            self.source = source
            self.inner = iterator
            self.kwargs = kwargs

        def __len__(self):
            return len(self.inner)

    def __init__(
        self,
        url: str,
        input: Union[
            np.ndarray, str, PathLike, ImageInfo, VideoInfo, ProjectInfo, DatasetInfo, list
        ] = None,
        image_id: int = None,
        image_ids: List[int] = None,
        video_id: int = None,
        dataset_id: int = None,
        project_id: int = None,
        params: dict = None,
        api: "Api" = None,
        **kwargs: dict,
    ):
        assert (
            sum(
                [
                    x is not None
                    for x in [input, image_id, image_ids, video_id, dataset_id, project_id]
                ]
            )
            == 1
        ), "Exactly one of input, image_id, image_ids, video_id, dataset_id, or project_id must be provided."

        self._session = None
        self._iterator = None
        self._base_url = url
        self.input = input
        self.params = params
        self.api = api

        if not isinstance(input, list):
            input = [input]
        if len(input) == 0:
            raise ValueError("Input cannot be empty.")
        if isinstance(input[0], np.ndarray):
            # input is numpy array
            kwargs = get_valid_kwargs(
                kwargs, self.session.inference_images_np_async, exclude=["images"]
            )
            self._iterator = self._Iterator(
                self.input, self.session.inference_images_np_async(input, **kwargs)
            )
        elif isinstance(input[0], (str, PathLike)):
            if len(input) > 1:
                # if the input is a list of paths, assume they are images
                for x in input:
                    if not isinstance(x, (str, PathLike)):
                        raise ValueError("Input must be a list of strings or PathLike objects.")
                kwargs = get_valid_kwargs(
                    kwargs, self.session.inference_image_paths_async, exclude=["image_paths"]
                )
                self._iterator = self._Iterator(
                    self.input,
                    self.session.inference_image_paths_async(input, **kwargs),
                )
            else:
                if dir_exists(input[0]):
                    # if the input is a directory, assume it contains images
                    kwargs = get_valid_kwargs(
                        kwargs, self.session.inference_image_paths_async, exclude=["image_paths"]
                    )
                    image_paths = list_files(input[0])
                    if len(image_paths) == 0:
                        raise ValueError("Directory is empty.")
                    self._iterator = self._Iterator(
                        self.input,
                        self.session.inference_image_paths_async(image_paths, **kwargs),
                    )
                elif file_exists(input[0]):
                    ext = get_file_ext(input[0])
                    if ext == "":
                        raise ValueError("File has no extension.")
                    if ext in SUPPORTED_IMG_EXTS:
                        kwargs = get_valid_kwargs(
                            kwargs,
                            self.session.inference_image_paths_async,
                            exclude=["image_paths"],
                        )
                        self._iterator = self._Iterator(
                            self.input,
                            self.session.inference_image_paths_async([input[0]], **kwargs),
                        )
                    elif ext in ALLOWED_VIDEO_EXTENSIONS:
                        kwargs = get_valid_kwargs(
                            kwargs, self.session.inference_video_path_async, exclude=["video_path"]
                        )
                        self._iterator = self._Iterator(
                            self.input,
                            self.session.inference_video_path_async(input[0], **kwargs),
                        )
                    else:
                        raise ValueError(
                            f"Unsupported file extension: {ext}. Supported extensions are: {SUPPORTED_IMG_EXTS + ALLOWED_VIDEO_EXTENSIONS}"
                        )
                else:
                    raise ValueError(f"File or directory does not exist: {input[0]}")

        elif isinstance(input[0], ImageInfo):
            kwargs = get_valid_kwargs(
                kwargs, self.session.inference_image_ids_async, exclude=["image_ids"]
            )
            self._iterator = self._Iterator(
                self.input,
                self.session.inference_image_ids_async([image.id for image in input], **kwargs),
            )
        elif isinstance(input[0], VideoInfo):
            if len(input) > 1:
                raise ValueError("Inference on multiple videos is not supported.")
            kwargs = get_valid_kwargs(
                kwargs, self.session.inference_video_id_async, exclude=["video_id"]
            )
            self._iterator = self._Iterator(
                self.input,
                self.session.inference_video_id_async(input[0].id, **kwargs),
            )
        elif isinstance(input[0], ProjectInfo):
            if len(input) > 1:
                raise ValueError("Inference on multiple projects is not supported.")
            kwargs = get_valid_kwargs(
                kwargs, self.session.inference_project_id_async, exclude=["project_id"]
            )
            self._iterator = self._Iterator(
                self.input,
                self.session.inference_project_id_async(input[0].id, **kwargs),
            )
        elif isinstance(input[0], DatasetInfo):
            if len(input) > 1:
                raise ValueError("Inference on multiple datasets is not supported.")
            kwargs = get_valid_kwargs(
                kwargs,
                self.session.inference_project_id_async,
                exclude=["project_id", "dataset_ids"],
            )
            dataset_ids = [input[0].id]
            project_id = input[0].project_id
            self._iterator = self._Iterator(
                self.input,
                self.session.inference_project_id_async(project_id, dataset_ids, **kwargs),
            )
        elif image_id is not None:
            kwargs = get_valid_kwargs(
                kwargs, self.session.inference_image_ids_async, exclude=["image_ids"]
            )
            self._iterator = self._Iterator(
                image_id, self.session.inference_image_ids_async([image_id], **kwargs)
            )
        elif image_ids is not None:
            kwargs = get_valid_kwargs(
                kwargs, self.session.inference_image_ids_async, exclude=["image_ids"]
            )
            self._iterator = self._Iterator(
                image_ids, self.session.inference_image_ids_async(image_ids, **kwargs)
            )
        elif video_id is not None:
            kwargs = get_valid_kwargs(
                kwargs, self.session.inference_video_id_async, exclude=["video_id"]
            )
            self._iterator = self._Iterator(
                video_id, self.session.inference_video_id_async(video_id, **kwargs)
            )
        elif dataset_id is not None:
            kwargs = get_valid_kwargs(
                kwargs, self.session.inference_project_id_async, exclude=["project_id"]
            )
            self._iterator = self._Iterator(
                dataset_id,
                self.session.inference_project_id_async(dataset_id, **kwargs),
            )
        elif project_id is not None:
            kwargs = get_valid_kwargs(
                kwargs, self.session.inference_project_id_async, exclude=["project_id"]
            )
            self._iterator = self._Iterator(
                project_id,
                self.session.inference_project_id_async(project_id, **kwargs),
            )
        else:
            raise ValueError(
                "Unknown input type. Supported types are: numpy array, path to a file or directory, ImageInfo, VideoInfo, ProjectInfo, DatasetInfo."
            )

    @property
    def session(self):
        from supervisely.nn.inference.session import SessionJSON

        if self._session is None:
            self._session = SessionJSON(
                api=self.api, session_url=self._base_url, inference_settings=self.params
            )
        return self._session

    def stop(self):
        try:
            self.session.stop_async_inference()
            self.session._on_async_inference_end()
        except Exception as e:
            logger.warning("Failed to stop the session: %s", e, exc_info=True)

    def next(self):
        return self.__next__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.stop()
            return False

    def __next__(self):
        prediction_json = self._iterator.inner.__next__()
        prediction = PredictionDTO.from_json(
            {**self._iterator.kwargs, **prediction_json},
            source=self.input,
            model_meta=self.session.get_model_meta(),
        )
        return prediction

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iterator)


class ModelApi:
    def __init__(
        self, api: "Api" = None, deploy_id: int = None, url: str = None, params: dict = None
    ):
        assert not (
            deploy_id is None and url is None
        ), "Either `deploy_id` or `url` must be passed."
        assert (
            deploy_id is None or url is None
        ), "Either `deploy_id` or `url` must be passed (not both)."
        if deploy_id is not None:
            assert api is not None, "API must be provided if `deploy_id` is passed."

        self.api = api
        self.deploy_id = deploy_id
        self.url = url
        self.params = params

        if self.deploy_id is not None:
            task_info = self.api.task.get_info_by_id(self.deploy_id)
            self.url = f'{self.api.server_address}/net/{task_info["meta"]["sessionToken"]}'

    def predict_detached(
        self,
        input: Union[
            np.ndarray, str, PathLike, ImageInfo, VideoInfo, ProjectInfo, DatasetInfo, list
        ] = None,
        image_id: int = None,
        image_ids: List[int] = None,
        video_id: int = None,
        dataset_id: int = None,
        project_id: int = None,
        params: Dict = None,
        **kwargs,
    ) -> InferenceSession:
        if (
            sum(
                [
                    x is not None
                    for x in [input, image_id, image_ids, video_id, dataset_id, project_id]
                ]
            )
            != 1
        ):
            raise ValueError(
                "Exactly one of input, image_id, image_ids, video_id, dataset_id, or project_id must be provided."
            )
        return InferenceSession(
            self.url,
            input=input,
            image_id=image_id,
            image_ids=image_ids,
            video_id=video_id,
            dataset_id=dataset_id,
            project_id=project_id,
            params=params,
            api=self.api,
            **kwargs,
        )

    def predict(
        self,
        input: Union[
            np.ndarray, str, PathLike, ImageInfo, VideoInfo, ProjectInfo, DatasetInfo, list
        ] = None,
        image_id: int = None,
        image_ids: List[int] = None,
        video_id: int = None,
        dataset_id: int = None,
        project_id: int = None,
        params: Dict = None,
        **kwargs,
    ) -> Union[PredictionDTO, List[PredictionDTO], InferenceSession]:

        session = self.predict_detached(
            input, image_id, image_ids, video_id, dataset_id, project_id, params, **kwargs
        )
        result = list(session)
        if isinstance(input, list):
            return result
        return result[0]

    def _post(self, method: str, data: dict, raise_for_status: bool = True):
        url = f"{self.url.rstrip("/")}/{method.lstrip("/")}"
        response = requests.post(url, json=data)
        if raise_for_status:
            response.raise_for_status()
        return response.json()

    def _get(self, method: str, params: dict = None, raise_for_status: bool = True):
        url = f"{self.url.rstrip('/')}/{method.lstrip('/')}"
        response = requests.get(url, params=params)
        if raise_for_status:
            response.raise_for_status()
        return response.json()

    def shutdown(self):
        if self.deploy_id is not None:
            return self.api.task.stop(self.deploy_id)
        response = self._post("tasks.stop", {ApiField.ID: id})
        return TaskApi.Status(response[ApiField.STATUS])

    def get_info(self):
        if self.deploy_id is not None:
            return self.api.nn._deploy_api.get_deploy_info(self.deploy_id)
        return self._post("get_deploy_info", {})

    def healthcheck(self):
        if self.deploy_id is not None:
            return self.api.task.is_ready(self.deploy_id)
        return self._post("is_ready", {})["status"] == "ready"

    def monitor(self):
        raise NotImplementedError
