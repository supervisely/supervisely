from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from supervisely.annotation.annotation import Annotation
from supervisely.api.api import Api
from supervisely.nn.inference.session import Session
from supervisely.video_annotation.video_annotation import VideoAnnotation


class InferenceSession:
    class Status:
        WAITING = "waiting"
        RUNNING = "running"
        FINISHED = "finished"
        FAILED = ""

    def __init__(self, url: str, params: dict = None):
        self._base_url = url
        self.params = params

        self.iterator = None
        self.status = self.Status.WAITING
        self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = Session(session_url=self._base_url)
        return self._session

    def start(
        self,
        image: Union[str, int, List[int], List[str]] = None,
        video: Union[str, int] = None,
        dataset: int = None,
        project: Union[str, int] = None,
    ) -> "InferenceSession":
        assert (
            sum([x is not None for x in [image, video, dataset, project]]) != 1
        ), "Exactly one of `image`, `video`, `project`, or `dataset` must be provided."
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            if isinstance(image[0], int):
                # image_ids
                self.iterator = self.session.inference_image_ids_async(image)
            else:
                # image_paths
                raise NotImplementedError()
                # iterator = self.session.inference_image_paths_async(image)
        else:  # video, dataset, or project
            raise NotImplementedError()
        self.status = self.Status.RUNNING
        return self

    def get(self):
        pass


@dataclass
class PredictionDTO:
    source: Union[str, int]
    annotation: Union[Annotation, VideoAnnotation]
    image_id: Optional[int]
    image_path: Optional[str]
    video_id: Optional[int]
    video_path: Optional[str]
    dataset_id: Optional[int]
    project_id: Optional[int]
    project_path: Optional[str]


class ModelApi:
    def __init__(
        self, api: Api = None, deploy_id: int = None, url: str = None, params: dict = None
    ):
        self._api = api
        assert not (
            deploy_id is None and url is None
        ), "Either `deploy_id` or `url` must be passed."
        assert (
            deploy_id is None or url is None
        ), "Either `deploy_id` or `url` must be passed (not both)."
        if deploy_id is not None:
            assert api is not None, "API must be provided if `deploy_id` is passed."

        self._deploy_id = deploy_id
        self._url = url
        self.params = params

    def predict(
        self,
        image: Union[str, int, List[int], List[str]] = None,
        video: Union[str, int] = None,
        dataset: int = None,
        project: Union[str, int] = None,
        params: Dict = None,
        detached: bool = False,
    ) -> Union[PredictionDTO, List[PredictionDTO], InferenceSession]:
        assert (
            sum([x is not None for x in [image, video, dataset, project]]) != 1
        ), "Exactly one of `image`, `video`, `project`, or `dataset` must be provided."

        session = InferenceSession(self._url, params=params)
        session.start(image=image, video=video, dataset=dataset, project=project)

        if detached:
            return session
        session = list(session)
        source = next(x for x in [image, video, dataset, project] if x is not None)
        if isinstance(source, list):
            return session
        return session[0]

    def shutdown(self):
        pass

    def get_info(self):
        pass

    def healthcheck(self):
        pass

    def monitor(self):
        pass
