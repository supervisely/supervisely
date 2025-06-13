import random
from pathlib import Path
from typing import List, Optional, Union

from supervisely.api.api import Api
from supervisely.app.widgets import SlyTqdm
from supervisely.imaging.color import get_predefined_colors, rgb2hex
from supervisely.io import env
from supervisely.io.fs import dir_empty, mkdir
from supervisely.io.json import load_json_file
from supervisely.nn.benchmark.comparison.detection_visualization.visualizer import (
    DetectionComparisonVisualizer,
)
from supervisely.nn.benchmark.comparison.semantic_segmentation.visualizer import (
    SemanticSegmentationComparisonVisualizer,
)
from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.nn.benchmark.object_detection.evaluator import (
    ObjectDetectionEvalResult,
)
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)
from supervisely.nn.task_type import TaskType
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly

ComparisonVisualizer = Union[
    DetectionComparisonVisualizer, SemanticSegmentationComparisonVisualizer
]
ComparisonEvalResult = Union[ObjectDetectionEvalResult, SemanticSegmentationEvalResult]


class ModelComparison:

    def __init__(
        self,
        api: Api,
        remote_eval_dirs: List[str],
        progress: Optional[SlyTqdm] = None,
        workdir: Optional[str] = "./benchmark/comparison",
        cv_task: Optional[TaskType] = None,
        team_id: Optional[int] = None,
    ):
        self.api = api
        self.progress = progress or tqdm_sly
        self.workdir = workdir
        self.remote_eval_dirs = remote_eval_dirs
        self.eval_results: List[ComparisonEvalResult] = []
        self.task_type = cv_task
        self.team_id = team_id or env.team_id()

        eval_cls = SemanticSegmentationEvalResult
        eval_cls = ObjectDetectionEvalResult

        colors = get_predefined_colors(len(remote_eval_dirs) * 5)  # for better visualizations
        random.shuffle(colors)
        for i, eval_dir in enumerate(remote_eval_dirs):
            local_path = Path(self.workdir) / "eval_data" / Path(eval_dir).name
            self._load_eval_data(eval_dir, str(local_path))

            eval_cls = self._get_eval_cls(str(local_path))
            eval_result = eval_cls(local_path / "evaluation")
            eval_result.report_path = Path(eval_dir, "visualizations", "template.vue").as_posix()
            eval_result.color = rgb2hex(colors[i])

            self.eval_results.append(eval_result)

        self._validate_eval_data()

        self.visualizer: ComparisonVisualizer = None
        self.remote_dir = None

    def _validate_eval_data(self):
        """
        Validate the evaluation data before running the comparison.
        Make sure the benchmarks are done on the same project and datasets.
        """
        task_type = None
        img_names = None
        cat_names = None
        for eval_result in self.eval_results:
            next_task_type = eval_result.cv_task
            if not task_type is None:
                assert task_type == next_task_type, "Task types are different in the evaluations."
            task_type = next_task_type
            if task_type == TaskType.SEMANTIC_SEGMENTATION:
                next_img_names = set(eval_result.mp.per_image_metrics.index)
            else:
                next_img_names = set(
                    [img.get("file_name") for img in eval_result.coco_gt.imgs.values()]
                )
            if not img_names is None:
                assert img_names == next_img_names, "Images are different in the evaluations."
            img_names = next_img_names
            if task_type == TaskType.SEMANTIC_SEGMENTATION:
                next_cat_names = set(eval_result.mp.class_names)
            else:
                next_cat_names = set(eval_result.mp.cat_names)
            if not cat_names is None:
                assert cat_names == next_cat_names, "Categories are different in the evaluations."
            cat_names = next_cat_names

    def visualize(self):
        task_type = self.eval_results[0].cv_task
        if task_type in [
            TaskType.OBJECT_DETECTION,
            TaskType.INSTANCE_SEGMENTATION,
        ]:
            vis_cls = DetectionComparisonVisualizer
        elif task_type == TaskType.SEMANTIC_SEGMENTATION:
            vis_cls = SemanticSegmentationComparisonVisualizer
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        if self.visualizer is None:
            self.visualizer = vis_cls(self)
        self.visualizer.visualize()

    def upload_results(self, team_id: int, remote_dir: str, progress=None) -> str:
        self.remote_dir = self.visualizer.upload_results(team_id, remote_dir, progress)
        return self.remote_dir

    def get_report_link(self) -> str:
        if self.remote_dir is None:
            raise ValueError("Results are not uploaded yet.")
        return self.visualizer.renderer._get_report_link(self.api, self.team_id, self.remote_dir)

    @property
    def report(self):
        return self.visualizer.renderer.report

    @property
    def lnk(self):
        return self.visualizer.renderer.lnk

    def _load_eval_data(self, src_path: str, dst_path: str) -> None:
        dir_name = Path(src_path).name
        if not dir_empty(dst_path):
            logger.info(f"Directory {dst_path} is not empty. Skipping download.")
            return
        if not self.api.storage.dir_exists(self.team_id, src_path):
            raise ValueError(f"Directory {src_path} not found in storage.")
        mkdir(dst_path)
        with self.progress(
            message=f"Downloading evaluation data from {dir_name}",
            total=self.api.storage.get_directory_size(self.team_id, src_path),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            self.api.storage.download_directory(
                self.team_id, src_path, dst_path, progress_cb=pbar.update
            )

    def _get_cv_task(self, eval_dir: str) -> CVTask:
        try:
            eval_data = load_json_file(Path(eval_dir, "evaluation", "inference_info.json"))
            task_type = eval_data.get("task_type")
            return CVTask(task_type.replace(" ", "_").lower())
        except Exception as e:
            raise ValueError(
                f"Could not get CV task from `inference_info.json`, try to set it manually. {e}"
            )

    def _get_eval_cls(self, eval_dir: str) -> ComparisonEvalResult:
        if self.task_type is None:
            self.task_type = self._get_cv_task(eval_dir)
        if self.task_type in [
            CVTask.OBJECT_DETECTION,
            CVTask.INSTANCE_SEGMENTATION,
        ]:
            eval_cls = ObjectDetectionEvalResult
        elif self.task_type == CVTask.SEMANTIC_SEGMENTATION:
            eval_cls = SemanticSegmentationEvalResult
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        return eval_cls
