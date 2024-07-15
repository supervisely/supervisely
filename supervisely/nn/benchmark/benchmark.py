from typing import Union, List

import supervisely as sly
from supervisely.nn.inference import SessionJSON, TaskType
from supervisely.nn.benchmark.evaluation import BaseEvaluator, ObjectDetectionEvaluator


class Benchmark:
    def __init__(
            self,
            api: sly.Api,
            model_session: Union[str, int, SessionJSON],
            inference_settings: dict = None,
            ):
        self.api = api
        self._model_session = model_session
        self._inference_settings = inference_settings
        self.session = self._init_model_session(api, model_session, inference_settings)
        self.cv_task = self._detect_cv_task()
        self.evaluator = self._get_evaluator()

    def run_evaluation(self):
        self.evaluator.run_evaluation()

    def run_speedtest(self, batch_sizes: List[int] = [1, 2, 4, 8, 16, 32, 64]):
        pass

    def upload_layout(self):
        pass

    def _detect_cv_task(self):
        return self.session.get_session_info()["task type"]

    def _get_evaluator(self) -> BaseEvaluator:
        if self.cv_task == TaskType.OBJECT_DETECTION:
            return ObjectDetectionEvaluator(self.api, self.session)
        else:
            raise NotImplementedError(f"Task type {self.cv_task} is not supported yet")

    def _init_model_session(
            self,
            api: sly.Api,
            model_session: Union[int, str, SessionJSON],
            inference_settings: dict = None
            ):
        if isinstance(model_session, int):
            session = SessionJSON(api, model_session)
        elif isinstance(model_session, str):
            session = SessionJSON(api, session_url=model_session)
        elif isinstance(model_session, SessionJSON):
            session = model_session
        else:
            raise ValueError(f"Unsupported type of 'model_session' argument: {type(model_session)}")
        
        if inference_settings is not None:
            session.set_inference_settings(inference_settings)
