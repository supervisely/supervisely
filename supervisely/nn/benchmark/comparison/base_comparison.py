import os
from typing import Optional, Tuple

from supervisely.api.api import Api
from supervisely.app.widgets import SlyTqdm
from supervisely.io.env import team_id
from supervisely.io.fs import dir_exists, mkdir
from supervisely.task.progress import tqdm_sly


class BaseComparison:
    def __init__(
        self,
        api: Api,
        eval_dir_1: str,
        eval_dir_2: str,
        progress: Optional[SlyTqdm] = None,
        output_dir: Optional[str] = "./comparison",  # ?
    ):
        self.api = api
        self.team_id = team_id()
        self._output_dir = output_dir
        self._eval_dir_1 = eval_dir_1
        self._eval_dir_2 = eval_dir_2
        self.pbar = progress or tqdm_sly

        self._load_eval_data()
        self.layout_dir = None

    def run_compare(self):
        raise NotImplementedError()

    def _load_eval_data(self):
        if dir_exists(self._eval_dir_1) and dir_exists(self._eval_dir_2):
            return
        dir_1_exists = self.api.storage.dir_exists(self.team_id, self._eval_dir_1)
        dir_2_exists = self.api.storage.dir_exists(self.team_id, self._eval_dir_2)
        if not dir_1_exists or not dir_2_exists:
            raise ValueError("One or both evaluation directories not found.")
        eval_dir_1, eval_dir_2 = self._get_eval_dirs()

        self.api.storage.download_directory(self.team_id, self._eval_dir_1, eval_dir_1)
        self.api.storage.download_directory(self.team_id, self._eval_dir_2, eval_dir_2)

        self._eval_dir_1 = eval_dir_1
        self._eval_dir_2 = eval_dir_2

    def _get_base_dir(self):
        base_dir = os.path.join(self.output_dir)
        if not dir_exists(base_dir):
            mkdir(base_dir)
        return base_dir

    def _get_eval_dirs(self) -> Tuple[str, str]:
        eval_dir_1 = os.path.join(self._get_base_dir(), "eval_1")
        if not dir_exists(eval_dir_1):
            mkdir(eval_dir_1)

        eval_dir_2 = os.path.join(self._get_base_dir(), "eval_2")
        if not dir_exists(eval_dir_2):
            mkdir(eval_dir_2)

        return eval_dir_1, eval_dir_2

    def _get_layout_dir(self):
        self.layout_dir = os.path.join(self._get_base_dir(), "layout")
        os.makedirs(self.layout_dir, exist_ok=True)
        return self.layout_dir
