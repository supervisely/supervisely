import inspect
from os import PathLike
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import numpy as np

from supervisely._utils import logger
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagValueType
from supervisely.api.annotation_api import AnnotationInfo
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.image_api import ImageInfo
from supervisely.api.project_api import ProjectInfo
from supervisely.api.video.video_api import VideoInfo
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.rectangle import Rectangle
from supervisely.project.project_meta import ProjectMeta

if TYPE_CHECKING:
    from supervisely.api.api import Api


def get_valid_kwargs(kwargs, func, exclude=None):
    signature = inspect.signature(func)
    valid_kwargs = {}
    for key, value in kwargs.items():
        if exclude is not None and key in exclude:
            continue
        if key in signature.parameters:
            valid_kwargs[key] = value
    return valid_kwargs


class PredictionDTO:

    def __init__(
        self,
        source: Union[str, int],
        annotation_json: Dict,
        model_meta: Optional[Union[ProjectMeta, Dict]] = None,
        image_id: Optional[int] = None,
        image_name: Optional[str] = None,
    ):
        self.source = source
        self.annotation_json = annotation_json

        self.model_meta = model_meta
        if isinstance(self.model_meta, dict):
            self.model_meta = ProjectMeta.from_json(self.model_meta)
        self.image_id = image_id
        self.image_name = image_name

        self._annotation = None
        self._boxes = None
        self._masks = None
        self._classes = None
        self._scores = None

    def _init_gemetries(self):

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
            self._init_gemetries()
        return self._boxes

    @property
    def masks(self):
        if self._masks is None:
            self._init_gemetries()
        return self._masks

    @property
    def classes(self):
        if self._classes is None:
            self._init_gemetries()
        return self._classes

    @property
    def scores(self):
        if self._scores is None:
            self._init_gemetries()
        return self._scores

    @property
    def annotation(self) -> Annotation:
        if self._annotation is None:
            self._annotation = Annotation.from_json(self.annotation_json, self.model_meta)
        return self._annotation


class InferenceSession:

    class _Iterator:
        def __init__(self, source: Any, iterator: Iterator):
            self.source = source
            self.inner = iterator

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

        source = input
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
                source, self.session.inference_images_np_async(input, **kwargs)
            )
        elif isinstance(input[0], (str, PathLike)):
            # input is path to a file or directory
            pass
        elif isinstance(input[0], ImageInfo):
            kwargs = get_valid_kwargs(
                kwargs, self.session.inference_image_ids_async, exclude=["image_ids"]
            )
            self._iterator = self._Iterator(
                source,
                self.session.inference_image_ids_async([image.id for image in input], **kwargs),
            )
        elif isinstance(input[0], VideoInfo):
            if len(input) > 1:
                raise ValueError("Inference on multiple videos is not supported.")
            kwargs = get_valid_kwargs(
                kwargs, self.session.inference_video_id_async, exclude=["video_id"]
            )
            self._iterator = self._Iterator(
                source,
                self.session.inference_video_id_async(input[0].id, **kwargs),
            )
        elif isinstance(input[0], ProjectInfo):
            if len(input) > 1:
                raise ValueError("Inference on multiple projects is not supported.")
            kwargs = get_valid_kwargs(
                kwargs, self.session.inference_project_id_async, exclude=["project_id"]
            )
            self._iterator = self._Iterator(
                source,
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
                source,
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
        kwargs = get_valid_kwargs(
            prediction_json,
            PredictionDTO.__init__,
            exclude=["self", "source", "annotation_json", "annotation", "model_meta"],
        )
        if "annotation" in prediction_json:
            annotation_json = prediction_json["annotation"]
        else:
            annotation_json = prediction_json
        prediction = PredictionDTO(
            source=self.input,
            annotation_json=annotation_json,
            model_meta=self.session.get_model_meta(),
            **kwargs,
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

    def shutdown(self):
        pass

    def get_info(self):
        pass

    def healthcheck(self):
        pass

    def monitor(self):
        pass
