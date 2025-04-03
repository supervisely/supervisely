import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

from supervisely._utils import logger
from supervisely.annotation.annotation import Annotation
from supervisely.api.annotation_api import AnnotationInfo
from supervisely.video_annotation.video_annotation import VideoAnnotation

if TYPE_CHECKING:
    from supervisely.api.api import Api


@dataclass
class PredictionDTO:
    source: Union[str, int]
    annotation: Union[Annotation, VideoAnnotation]
    image_id: Optional[int] = None
    image_path: Optional[str] = None
    video_id: Optional[int] = None
    video_path: Optional[str] = None
    dataset_id: Optional[int] = None
    project_id: Optional[int] = None
    project_path: Optional[str] = None


def get_valid_kwargs(kwargs, func, exclude=None):
    signature = inspect.signature(func)
    valid_kwargs = {}
    for key, value in kwargs.items():
        if exclude is not None and key in exclude:
            continue
        if key in signature.parameters:
            valid_kwargs[key] = value
    return valid_kwargs


class InferenceSession:

    class _Iterator:
        def __init__(self, source: Any, iterator: Iterator):
            self.source = source
            self.inner = iterator

    def __init__(
        self,
        url: str,
        images: Union[str, int, List[int], List[str]] = None,
        video: Union[str, int] = None,
        dataset: int = None,
        project: Union[str, int] = None,
        params: dict = None,
        **kwargs: dict,
    ):
        assert (
            sum([x is not None for x in [images, video, dataset, project]]) == 1
        ), "Exactly one of `images`, `video`, `project`, or `dataset` must be provided."

        self._session = None
        self._iterator = None
        self._base_url = url
        self.images = images
        self.video = video
        self.dataset = dataset
        self.project = project
        self.params = params

        if images is not None:
            source = images
            if not isinstance(images, list):
                images = [images]
            if isinstance(images[0], int):
                # image_ids
                kwargs = get_valid_kwargs(
                    kwargs, self.session.inference_image_ids_async, exclude=["image_ids"]
                )
                self._iterator = self._Iterator(
                    source,
                    self.session.inference_image_ids_async(images, **kwargs),
                )
            else:
                kwargs = get_valid_kwargs(
                    kwargs, self.session.inference_image_paths_async, exclude=["image_paths"]
                )
                self._iterator = self._Iterator(
                    source, self.session.inference_image_paths_async(images, **kwargs)
                )
        elif video is not None:
            if isinstance(video, int):
                kwargs = get_valid_kwargs(
                    kwargs, self.session.inference_video_id_async, exclude=["video_id", "source"]
                )
                self._iterator = self._Iterator(
                    video, self.session.inference_video_id_async(video, **kwargs)
                )
            else:
                kwargs = get_valid_kwargs(
                    kwargs,
                    self.session.inference_video_path_async,
                    exclude=["video_path", "source"],
                )
                self._iterator = self._Iterator(
                    video, self.session.inference_video_path_async(video, **kwargs)
                )
        else:  # dataset or project
            raise NotImplementedError()

    @property
    def session(self):
        from supervisely.nn.inference.session import Session

        if self._session is None:
            self._session = Session(session_url=self._base_url, inference_settings=self.params)
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
        kwargs = {}
        annotation = self._iterator.inner.__next__()
        if isinstance(annotation, AnnotationInfo):
            kwargs["image_id"] = annotation.image_id
            kwargs["dataset_id"] = annotation.dataset_id
            annotation = annotation.annotation
        prediction = PredictionDTO(source=self.images, annotation=annotation, **kwargs)
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
        images: Union[str, int, List[int], List[str]] = None,
        video: Union[str, int] = None,
        dataset: int = None,
        project: Union[str, int] = None,
        params: Dict = None,
        **kwargs,
    ) -> InferenceSession:
        return InferenceSession(
            self.url,
            images=images,
            video=video,
            dataset=dataset,
            project=project,
            params=params,
            **kwargs,
        )

    def predict(
        self,
        images: Union[str, int, List[int], List[str]] = None,
        video: Union[str, int] = None,
        dataset: int = None,
        project: Union[str, int] = None,
        params: Dict = None,
        **kwargs,
    ) -> Union[PredictionDTO, List[PredictionDTO], InferenceSession]:

        session = self.predict_detached(images, video, dataset, project, params, **kwargs)
        result = list(session)
        source = next(x for x in [images, video, dataset, project] if x is not None)
        if isinstance(source, list):
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
