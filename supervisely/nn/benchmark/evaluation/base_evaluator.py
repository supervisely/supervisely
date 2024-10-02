import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from supervisely import logger
from supervisely.app.widgets import SlyTqdm
from supervisely.task.progress import tqdm_sly


class BaseEvaluator:

    def __init__(
        self,
        gt_project_path: str,
        dt_project_path: str,
        result_dir: str = "./evaluation",
        progress: Optional[SlyTqdm] = None,
        items_count: Optional[int] = None,  # TODO: is it needed?
        classes_whitelist: Optional[List[str]] = None,
        parameters: Optional[Union[Dict, str]] = None,
    ):
        self.gt_project_path = gt_project_path
        self.dt_project_path = dt_project_path
        self.result_dir = result_dir
        self.total_items = items_count
        self.pbar = progress or tqdm_sly
        os.makedirs(result_dir, exist_ok=True)
        self.classes_whitelist = classes_whitelist
        self.parameters = self._read_parameters(parameters)

    def _read_parameters(self, parameters: Union[str, Dict]) -> Dict:
        if isinstance(parameters, str):
            try:
                if os.path.exists(parameters):
                    with open(parameters, "r") as f:
                        return yaml.safe_load(f)
                else:
                    return yaml.safe_load(parameters)
            except Exception as e:
                logger.warning(
                    f"Failed to load evaluation parameters: {e}. Using default parameters."
                )
                return None
        return parameters

    @staticmethod
    def default_parameters() -> str:
        with open(f"{Path(__file__).parent}/default_parameters.yml", "r") as f:
            return f.read()

    def evaluate(self):
        raise NotImplementedError()

    def get_result_dir(self) -> str:
        return self.result_dir

    def _dump_pickle(self, data, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
