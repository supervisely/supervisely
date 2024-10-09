import os
import pickle
from typing import List, Optional

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
    ):
        self.gt_project_path = gt_project_path
        self.dt_project_path = dt_project_path
        self.result_dir = result_dir
        self.total_items = items_count
        self.pbar = progress or tqdm_sly
        os.makedirs(result_dir, exist_ok=True)
        self.classes_whitelist = classes_whitelist

    def evaluate(self):
        raise NotImplementedError()

    def get_result_dir(self) -> str:
        return self.result_dir

    def _dump_pickle(self, data, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
