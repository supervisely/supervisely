import random
from pathlib import Path
from typing import List, Optional

from supervisely.api.api import Api
from supervisely.app.widgets import SlyTqdm
from supervisely.imaging.color import get_predefined_colors, rgb2hex
from supervisely.nn.benchmark.comparison.detection_visualization.visualizer import (
    DetectionComparisonVisualizer,
)
from supervisely.nn.benchmark.visualization.evaluation_result import EvalResult
from supervisely.task.progress import tqdm_sly


class ModelComparison:

    def __init__(
        self,
        api: Api,
        remote_eval_dirs: List[str],
        progress: Optional[SlyTqdm] = None,
        workdir: Optional[str] = "./benchmark/comparison",
    ):
        self.api = api
        self.progress = progress or tqdm_sly
        self.workdir = workdir
        self.remote_eval_dirs = remote_eval_dirs
        self.evaluation_results: List[EvalResult] = []

        colors = get_predefined_colors(len(remote_eval_dirs) * 5)  # for better visualizations
        random.shuffle(colors)
        for i, eval_dir in enumerate(remote_eval_dirs):
            local_path = str(Path(self.workdir, "eval_data"))
            eval_result = EvalResult(eval_dir, local_path, self.api, self.progress)
            self.evaluation_results.append(eval_result)
            eval_result.color = rgb2hex(colors[i])

        self.task_type = self.evaluation_results[0].inference_info.get("task_type")
        self._validate_eval_data()

        self.visualizer: DetectionComparisonVisualizer = None
        self.remote_dir = None

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
            self.visualizer = DetectionComparisonVisualizer(self)
        self.visualizer.visualize()

    def upload_results(self, team_id: int, remote_dir: str, progress=None) -> str:
        self.remote_dir = self.visualizer.upload_results(team_id, remote_dir, progress)
        return self.remote_dir

    def get_report_link(self) -> str:
        if self.remote_dir is None:
            raise ValueError("Results are not uploaded yet.")
        report_link = self.remote_dir.rstrip("/") + "/template.vue"
        return report_link
