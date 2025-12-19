from __future__ import annotations

import json
import os
import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from supervisely.app.widgets import SlyTqdm
from supervisely.io.fs import get_file_name_with_ext, silent_remove
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.task.progress import tqdm_sly


class BaseEvalResult:
    PRIMARY_METRIC = None

    def __init__(self, directory: Optional[str] = None):
        self.directory = directory
        self.inference_info: Dict = {}
        self.speedtest_info: Optional[Dict] = None
        self.eval_data: Optional[Dict] = None
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
    def short_name(self) -> Union[str, None]:
        if not self.name:
            return None
        if len(self.name) > 20:
            return self.name[:9] + "..." + self.name[-7:]
        return self.name

    @property
    def gt_project_id(self) -> Optional[int]:
        return self.inference_info.get("gt_project_id")

    @property
    def gt_dataset_ids(self) -> Optional[List[int]]:
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

    def _load_eval_data_archive(self, path: Path, pd_index_col=False) -> Dict:
        """Load eval_data from archive"""
        with zipfile.ZipFile(path, mode="r") as zf:
            with zf.open("eval_data.json") as json_f:
                data = json.load(json_f)
            return self._process_value_from_archive(data, zf, pd_index_col)

    def _process_value_from_archive(self, value, zf: zipfile.ZipFile, pd_index_col: bool = False):
        """Recursively process values from archive, handling nested dicts and lists."""
        if isinstance(value, str) and value.endswith(".npy"):
            with zf.open(value) as arr_f:
                return np.load(arr_f)
        elif isinstance(value, str) and value.endswith(".csv"):
            with zf.open(value) as df_f:
                if pd_index_col:
                    return pd.read_csv(df_f, sep="\t", index_col=0)
                return pd.read_csv(df_f, sep="\t")
        elif isinstance(value, dict):
            res = {}
            for k, v in value.items():
                k = int(k) if isinstance(k, str) and k.isdigit() else k
                k = float(k) if isinstance(k, str) and self._is_float(k) else k
                res[k] = self._process_value_from_archive(v, zf, pd_index_col)
            return res
        elif isinstance(value, list):
            return [self._process_value_from_archive(item, zf, pd_index_col) for item in value]
        elif isinstance(value, str) and value.isdigit():
            return int(value)
        else:
            return value

    def _is_float(self, s: str) -> bool:
        if not s or not isinstance(s, str):
            return False
        try:
            float(s)
            return "." in s or "e" in s.lower()
        except (ValueError, AttributeError):
            return False


class BaseEvaluator:
    EVALUATION_PARAMS_YAML_PATH: Optional[str] = None
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
        self.eval_data: Optional[Dict] = None
        self.gt_project_path = gt_project_path
        self.pred_project_path = pred_project_path
        self.result_dir = result_dir
        self.total_items = items_count
        self.pbar = progress or tqdm_sly
        os.makedirs(result_dir, exist_ok=True)
        self.classes_whitelist = classes_whitelist or []

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
            params = cls.load_yaml_evaluation_params()
            if params is None:
                return {}
            return yaml.safe_load(params)

    def _dump_pickle(self, data, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def _process_value_for_archive(self, value: Any, key_prefix: str, zf: zipfile.ZipFile) -> Any:
        """Recursively process values for archiving, handling nested dicts and lists."""
        if isinstance(value, np.ndarray):
            filename = f"{key_prefix}.npy" if key_prefix else "array.npy"
            filepath = os.path.join(self.result_dir, filename)
            np.save(filepath, value)
            zf.write(filepath, arcname=filename)
            os.remove(filepath)
            return filename
        elif isinstance(value, pd.DataFrame):
            filename = f"{key_prefix}.csv" if key_prefix else "dataframe.csv"
            filepath = os.path.join(self.result_dir, filename)
            value.to_csv(filepath, sep="\t")
            zf.write(filepath, arcname=filename)
            os.remove(filepath)
            return filename
        elif isinstance(value, dict):
            return {
                k: self._process_value_for_archive(v, f"{key_prefix}.{k}" if key_prefix else k, zf)
                for k, v in value.items()
            }
        elif isinstance(value, list):
            return [
                self._process_value_for_archive(item, f"{key_prefix}[{i}]", zf)
                for i, item in enumerate(value)
            ]
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, str) and value.isdigit():
            return int(value)
        else:
            return value

    def _dump_eval_results_archive(self):
        with zipfile.ZipFile(os.path.join(self.result_dir, "eval_data.zip"), mode="w") as zf:
            data = self._process_value_for_archive(self.eval_data, "", zf)
            filepath = os.path.join(self.result_dir, "eval_data.json")
            dump_json_file(data, filepath, indent=4)
            zf.write(filepath, arcname="eval_data.json")
            silent_remove(filepath)

    def get_eval_result(self) -> BaseEvalResult:
        return self.eval_result_cls(self.result_dir)
