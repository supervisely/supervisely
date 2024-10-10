from typing import Dict, List, Optional

from supervisely.api.api import Api
from supervisely.app.widgets import SlyTqdm
from supervisely.nn.benchmark.comparison.evaluation_result import EvalResult
from supervisely.nn.benchmark.comparison.visualization.visualizer import (
    ComparisonVisualizer,
)
from supervisely.task.progress import tqdm_sly


class BaseComparison:

    def __init__(
        self,
        api: Api,
        remote_eval_dirs: List[str],
        progress: Optional[SlyTqdm] = None,
        output_dir: Optional[str] = "./benchmark/comparison",
    ):
        self.api = api
        self.progress = progress or tqdm_sly
        self.output_dir = output_dir
        self.remote_eval_dirs = remote_eval_dirs
        self.evaluation_results: List[EvalResult] = []
        for eval_dir in remote_eval_dirs:
            self.evaluation_results.append(
                EvalResult(eval_dir, self.output_dir, self.api, self.progress)
            )

        self.task_type = self.evaluation_results[0].inference_info.get("task_type")
        self._validate_eval_data()

        self.visualizer: ComparisonVisualizer = None

    def run_compare(self):
        raise NotImplementedError()

    def _validate_eval_data(self):
        """
        Validate the evaluation data before running the comparison.
        Make sure the benchmarks are done on the same project and datasets.
        """
        task_type = None
        img_names = None
        cat_names = None
        for eval_result in self.evaluation_results:
            next_task_type = eval_result.cv_task
            if not task_type is None:
                assert task_type == next_task_type, "Task types are different in the evaluations."
            task_type = next_task_type
            next_img_names = set(
                [img.get("file_name") for img in eval_result.coco_gt.imgs.values()]
            )
            if not img_names is None:
                assert img_names == next_img_names, "Images are different in the evaluations."
            img_names = next_img_names
            next_cat_names = set([cat.get("name") for cat in eval_result.coco_gt.cats.values()])
            if not cat_names is None:
                assert cat_names == next_cat_names, "Categories are different in the evaluations."
            cat_names = next_cat_names

    def get_metrics(self):
        pass

    def visualize(self):
        if self.visualizer is None:
            self.visualizer = ComparisonVisualizer(self)
        self.visualizer.visualize()

    def upload_results(self, team_id: int, remote_dir: str) -> str:
        return self.visualizer.upload_results(team_id, remote_dir)

    def get_report_link(self, team_id: int, remote_dir: str) -> str:
        return ""  # TODO: implement
