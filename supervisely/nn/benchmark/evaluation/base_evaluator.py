import os
import pickle
from typing import List, Optional, Union

import yaml

from supervisely.app.widgets import SlyTqdm
from supervisely.task.progress import tqdm_sly


class BaseEvaluator:
    EVALUATION_PARAMS_YAML_PATH: str = None

    def __init__(
        self,
        gt_project_path: str,
        dt_project_path: str,
        result_dir: str = "./evaluation",
        progress: Optional[SlyTqdm] = None,
        items_count: Optional[int] = None,  # TODO: is it needed?
        classes_whitelist: Optional[List[str]] = None,
        evaluation_params: Optional[dict] = None,
    ):
        self.gt_project_path = gt_project_path
        self.dt_project_path = dt_project_path
        self.result_dir = result_dir
        self.total_items = items_count
        self.pbar = progress or tqdm_sly
        os.makedirs(result_dir, exist_ok=True)
        self.classes_whitelist = classes_whitelist

        if evaluation_params is None:
            evaluation_params = self._get_default_evaluation_params()
        self.evaluation_params = evaluation_params
        if self.evaluation_params:
            self.validate_evaluation_params(self.evaluation_params)

    def evaluate(self):
        raise NotImplementedError()

    def get_result_dir(self) -> str:
        return self.result_dir

    @classmethod
    def load_yaml_evaluation_params(cls) -> Union[str, None]:
        if cls.EVALUATION_PARAMS_YAML_PATH is None:
            return None
        with open(cls.EVALUATION_PARAMS_YAML_PATH, "r") as f:
            return f.read()

    @classmethod
    def validate_evaluation_params(cls, evaluation_params: dict) -> None:
        pass

    @classmethod
    def _get_default_evaluation_params(cls) -> dict:
        if cls.EVALUATION_PARAMS_YAML_PATH is None:
            return {}
        else:
            return yaml.safe_load(cls.load_yaml_evaluation_params())

    def _dump_pickle(self, data, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
