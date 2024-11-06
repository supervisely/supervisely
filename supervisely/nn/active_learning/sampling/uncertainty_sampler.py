from typing import Union
from supervisely.nn.inference import SessionJSON, Session
from supervisely.nn.active_learning.sampling.base_sampler import BaseSampler
from supervisely.nn.benchmark.utils import try_set_conf_auto  # TODO: move out of benchmark
from supervisely import Api
from supervisely import logger


class UncertaintySampler(BaseSampler):
    def __init__(
        self,
        api: Api,
        project_id: int,
        image_ids: list,
        model_session: Union[int, str, SessionJSON],
        confidence_range=(0.4, 0.6),
    ):
        super().__init__(api, project_id, image_ids)
        self.confidence_range = confidence_range
        self.model_session = self._init_model_session(model_session)
        assert try_set_conf_auto(
            self.model_session, confidence_range[0]
        ), f"Unable to set the confidence threshold to {confidence_range[0]} for evalation."

    def sample(self, num_images: int):
        sampled_image_ids = []
        iterator = self.model_session.inference_project_id_async(self.project_id)
        i = 0
        for predict in iterator:
            print(i)
            i += 1
            ann = predict["annotation"]
            for obj in ann["objects"]:
                conf = [tag["value"] for tag in obj["tags"] if tag["name"] == "confidence"][0]
                if self.confidence_range[0] <= conf <= self.confidence_range[1]:
                    sampled_image_ids.append(predict["image_id"])
                    break
            if len(sampled_image_ids) >= num_images:
                self.model_session.stop_async_inference()
                self.model_session._on_async_inference_end()
                break
        return sampled_image_ids

    def _init_model_session(
        self, model_session: Union[int, str, SessionJSON], inference_settings: dict = None
    ):
        if isinstance(model_session, int):
            session = SessionJSON(self.api, model_session)
        elif isinstance(model_session, str):
            session = SessionJSON(self.api, session_url=model_session)
        elif isinstance(model_session, Session):
            session = SessionJSON(model_session.api, session_url=model_session._base_url)
        elif isinstance(model_session, SessionJSON):
            session = model_session
        else:
            raise ValueError(f"Unsupported type of 'model_session' argument: {type(model_session)}")

        if inference_settings is not None:
            session.set_inference_settings(inference_settings)
        return session
