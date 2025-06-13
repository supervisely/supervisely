from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Union

import yaml

from supervisely.app.widgets import SlyTqdm
from supervisely.io.fs import get_file_name_with_ext
from supervisely.task.progress import tqdm_sly


class BaseEvalResult:
    PRIMARY_METRIC = None

    def __init__(self, directory: Optional[str] = None):
        self.directory = directory
        self.inference_info: Dict = {}
        self.speedtest_info: Dict = None
        self.eval_data: Dict = None
        self.mp = None

        if self.directory is not None:
            self._read_files(self.directory)
            self._prepare_data()

    @classmethod
    def from_evaluator(cls, evaulator: BaseEvaluator) -> BaseEvalResult:
        """Method to customize loading of the evaluation result."""
        raise NotImplementedError()

    @property
    def cv_task(self):
        return self.inference_info.get("task_type")

    @property
    def name(self) -> Union[str, None]:
        deploy_params = self.inference_info.get("deploy_params", {})
        return (
            deploy_params.get("checkpoint_name")
            or deploy_params.get("model_name")
            or self.inference_info.get("model_name")
        )

    @property
    def short_name(self) -> str:
        if not self.name:
            return
        if len(self.name) > 20:
            return self.name[:9] + "..." + self.name[-7:]
        return self.name

    @property
    def gt_project_id(self) -> int:
        return self.inference_info.get("gt_project_id")

    @property
    def gt_dataset_ids(self) -> List[int]:
        return self.inference_info.get("gt_dataset_ids", None)

    @property
    def dt_project_id(self):
        return self.inference_info.get("dt_project_id")

    @property
    def pred_project_id(self):
        return self.dt_project_id

    @property
    def train_info(self):
        return self.inference_info.get("train_info", None)  # TODO: check

    @property
    def evaluator_app_info(self):
        return self.inference_info.get("evaluator_app_info", None)  # TODO: check

    @property
    def val_images_cnt(self):
        return self.inference_info.get("val_images_cnt", None)  # TODO: check

    @property
    def classes_whitelist(self):
        return self.inference_info.get("inference_settings", {}).get("classes", [])  # TODO: check

    def _read_files(self, path: str) -> None:
        """Read all necessary files from the directory"""
        raise NotImplementedError()

    def _prepare_data(self) -> None:
        """Prepare data to allow easy access to the data"""
        raise NotImplementedError()

    @property
    def key_metrics(self):
        raise NotImplementedError()

    @property
    def checkpoint_name(self):
        if self.inference_info is None:
            return None

        deploy_params = self.inference_info.get("deploy_params", {})
        name = None
        if deploy_params:
            name = deploy_params.get("checkpoint_name")  # not TrainApp
            if name is None:
                name = deploy_params.get("model_files", {}).get("checkpoint")
                if name is not None:
                    name = get_file_name_with_ext(name)
        if name is None:
            name = self.inference_info.get("checkpoint_name", "")
        return name


class BaseEvaluator:
    EVALUATION_PARAMS_YAML_PATH: str = None
    eval_result_cls = BaseEvalResult

    def __init__(
        self,
        gt_project_path: str,
        pred_project_path: str,
        result_dir: str = "./evaluation",
        progress: Optional[SlyTqdm] = None,
        items_count: Optional[int] = None,  # TODO: is it needed?
        classes_whitelist: Optional[List[str]] = None,
        evaluation_params: Optional[dict] = None,
    ):
        self.gt_project_path = gt_project_path
        self.pred_project_path = pred_project_path
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

    def get_eval_result(self) -> BaseEvalResult:
        return self.eval_result_cls(self.result_dir)
