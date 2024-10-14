import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import SlyTqdm
from supervisely.io.env import team_id
from supervisely.io.fs import dir_empty, mkdir
from supervisely.io.json import load_json_file
from supervisely.nn.benchmark.coco_utils import read_coco_datasets
from supervisely.nn.benchmark.evaluation.coco.metric_provider import MetricProvider
from supervisely.task.progress import tqdm_sly


class EvalResult:

    def __init__(
        self,
        eval_dir: str,
        output_dir: str,
        api: Api,
        progress: Optional[SlyTqdm] = None,
    ):
        from pycocotools.coco import COCO  # pylint: disable=import-error

        self.eval_dir = eval_dir
        self.output_dir = output_dir
        self.api = api
        self.team_id = team_id()
        self.local_dir = str(Path(self.output_dir, "eval_data", self.eval_dir.lstrip("/")))
        self.progress = progress or tqdm_sly

        self.coco_gt: COCO = None
        self.coco_dt: COCO = None
        self.inference_info: Dict = None
        self.eval_data: Dict = None
        self.mp: MetricProvider = None
        self.df_score_profile: pd.DataFrame = None
        self.dfsp_down: pd.DataFrame = None

        self._gt_project_info = None
        self._gt_dataset_infos = None

        self._load_eval_data()
        self._read_eval_data()

    @property
    def cv_task(self):
        return self.inference_info.get("task_type")

    @property
    def name(self) -> str:
        model_name = self.inference_info.get("model_name", self.eval_dir)
        return self.inference_info.get("deploy_params", {}).get("checkpoint_name", model_name)

    @property
    def gt_project_id(self) -> int:
        return self.inference_info.get("gt_project_id")

    @property
    def gt_project_info(self) -> ProjectInfo:
        if self._gt_project_info is None:
            gt_project_id = self.inference_info.get("gt_project_id")
            self._gt_project_info = self.api.project.get_info_by_id(gt_project_id)
        return self._gt_project_info

    @property
    def gt_dataset_ids(self) -> List[int]:
        return self.inference_info.get("gt_dataset_ids", None)

    @property
    def gt_dataset_infos(self) -> List[DatasetInfo]:
        if self.gt_dataset_ids is None:
            return None
        if self._gt_dataset_infos is None:
            self._gt_dataset_infos = self.api.dataset.get_list(
                self.gt_project_id,
                filters=[{"field": "id", "operator": "in", "value": self.gt_dataset_ids}],
                recursive=True,
            )
        return self._gt_dataset_infos

    @property
    def train_info(self):
        return self.inference_info.get("train_info", None)  # TODO: check

    @property
    def gt_images_ids(self):
        return self.inference_info.get("gt_images_ids", None)  # TODO: check

    @property
    def classes_whitelist(self):
        return self.inference_info.get("inference_settings", {}).get("classes", [])  # TODO: check

    def _load_eval_data(self):
        if not dir_empty(self.local_dir):
            return
        if not self.api.storage.dir_exists(self.team_id, self.eval_dir):
            raise ValueError(f"Directory {self.eval_dir} not found in storage.")
        mkdir(self.local_dir)
        with self.progress(
            message=f"Downloading evaluation data at {self.eval_dir}",
            total=self.api.storage.get_directory_size(self.team_id, self.eval_dir),
        ) as pbar:
            self.api.storage.download_directory(
                self.team_id, self.eval_dir, self.local_dir, progress_cb=pbar.update
            )

    def _read_eval_data(self):
        gt_path = Path(self.local_dir, "evaluation", "cocoGt.json")
        dt_path = Path(self.local_dir, "evaluation", "cocoDt.json")
        coco_gt, coco_dt = read_coco_datasets(load_json_file(gt_path), load_json_file(dt_path))
        self.coco_gt = coco_gt
        self.coco_dt = coco_dt
        self.eval_data = pickle.load(
            open(Path(self.local_dir, "evaluation", "eval_data.pkl"), "rb")
        )
        self.inference_info = load_json_file(
            Path(self.local_dir, "evaluation", "inference_info.json")
        )

        self.mp = MetricProvider(
            self.eval_data["matches"],
            self.eval_data["coco_metrics"],
            self.eval_data["params"],
            self.coco_gt,
            self.coco_dt,
        )
        self.mp.calculate()

        self.df_score_profile = pd.DataFrame(
            self.mp.confidence_score_profile(), columns=["scores", "precision", "recall", "f1"]
        )

        # downsample
        if len(self.df_score_profile) > 5000:
            self.dfsp_down = self.df_score_profile.iloc[:: len(self.df_score_profile) // 1000]
        else:
            self.dfsp_down = self.df_score_profile
