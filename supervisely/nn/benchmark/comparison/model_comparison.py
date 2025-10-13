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

            self._check_project_existence(eval_result)
            self.eval_results.append(eval_result)

        self.images_partially_matched = False
        self.matched_images_dict = {}
        self.classes_partially_matched = False
        self.matched_classes_dict = {}

        self._validate_task_type()
        self._get_overlapping_categories()
        self._get_overlapping_images()

        self.visualizer: ComparisonVisualizer = None
        self.remote_dir = None

    def _check_project_existence(self, eval_result):
        """
        Check if the ground truth project exists for the evaluation result.
        """
        if eval_result.gt_project_id is not None:
            eval_result.project_exists = True
            if not self.api.project.get_info_by_id(eval_result.gt_project_id):
                logger.warning(
                    f"Ground truth project with ID {eval_result.gt_project_id} not found."
                )
                eval_result.project_exists = False

        # if eval_result.pred_project_id is not None:
        #     eval_result.project_exists = True
        #     if not self.api.project.get_info_by_id(eval_result.pred_project_id):
        #         logger.warning(
        #             f"Prediction project with ID {eval_result.pred_project_id} not found."
        #         )
        #         eval_result.project_exists = False

    def _validate_task_type(self):
        """
        Validate the task type of the evaluations.
        Ensure that all evaluations are for the same task type.
        """
        task_type = None
        try:
            for eval_result in self.eval_results:
                next_task_type = eval_result.cv_task
                if task_type is not None:
                    assert task_type == next_task_type
                task_type = next_task_type
        except AssertionError:
            raise RuntimeError(
                f"Comparison validation failed: Task types are different in the evaluations"
            )

    def _get_overlapping_images(self):
        """
        Get the set of images that are present in all evaluations.
        This is used to ensure that the evaluations are comparable.
        """
        task_type = self.eval_results[0].cv_task
        if task_type == TaskType.SEMANTIC_SEGMENTATION:
            get_image_names = lambda eval_result: set(eval_result.mp.per_image_metrics.index)
        else:
            get_image_names = lambda eval_result: set(
                [img.get("file_name") for img in eval_result.coco_gt.imgs.values()]
            )

        image_names = get_image_names(self.eval_results[0])
        max_img_cnt = len(image_names)
        for eval_result in self.eval_results[1:]:
            current_image_names = get_image_names(eval_result)
            image_names.intersection_update(current_image_names)
            max_img_cnt = max(max_img_cnt, len(current_image_names))

        if not image_names:
            raise RuntimeError(
                "Comparison validation failed: No overlapping images found in the evaluations."
            )

        if len(image_names) == max_img_cnt:
            logger.info("All images are identical in all evaluations, no filtering applied.")
            return

        self.matched_images_dict = {
            "max": max_img_cnt,
            "current": len(image_names),
            "percentage": f"{len(image_names) / max_img_cnt * 100:.2f}%",
        }
        self.images_partially_matched = True
        logger.info(
            "Found %s overlapping images across evaluations",
            f"{self.matched_images_dict['current']}/{self.matched_images_dict['max']} ({self.matched_images_dict['percentage']})",
        )

        if task_type == TaskType.SEMANTIC_SEGMENTATION:
            for eval_result in self.eval_results:
                s = eval_result.mp.per_image_metrics["img_names"]
                eval_result.mp.per_image_metrics["img_names"] = [
                    img for img in s if img in image_names
                ]
                eval_result.images_map = dict(
                    filter(lambda item: item[0] in image_names, eval_result.images_map.items())
                )
                eval_result.matched_pair_data = dict(
                    filter(
                        lambda item: item[0] in image_names,
                        eval_result.matched_pair_data.items(),
                    )
                )
        else:
            for eval_result in self.eval_results:
                coco_gt_name_to_sly_id = {}
                coco_gt_name_to_id = {}
                for img in eval_result.coco_gt.imgs.values():
                    coco_gt_name_to_sly_id[img["file_name"]] = img["sly_id"]
                    coco_gt_name_to_id[img["file_name"]] = img["id"]

                coco_filtered_ids = [coco_gt_name_to_id[img_name] for img_name in image_names]
                # todo: check if necessary
                eval_result.coco_gt.anns = {
                    k: v
                    for k, v in eval_result.coco_gt.anns.items()
                    if v["image_id"] in coco_filtered_ids
                }

                eval_result.coco_gt.imgs = {
                    k: v
                    for k, v in eval_result.coco_gt.imgs.items()
                    if v["file_name"] in image_names
                }

                # outcome counts
                sly_ids_filtered = [coco_gt_name_to_sly_id[img_name] for img_name in image_names]
                for outcome, matches_data in eval_result.click_data.outcome_counts.items():
                    new_matches_data = [
                        match_data
                        for match_data in matches_data
                        if match_data["gt_img_id"] in sly_ids_filtered
                    ]
                    eval_result.click_data.outcome_counts[outcome] = new_matches_data

                # objects by class
                for cat_name, matches_data in eval_result.click_data.objects_by_class.items():
                    new_matches_data = [
                        match_data
                        for match_data in matches_data
                        if match_data["gt_img_id"] in image_names
                    ]
                    eval_result.click_data.objects_by_class[cat_name] = new_matches_data

        for eval_result in self.eval_results:
            eval_result.image_whitelist = list(image_names)

    def _get_overlapping_categories(self):
        """
        Get the set of categories that are present in all evaluations.
        This is used to ensure that the evaluations are comparable.
        """
        task_type = self.eval_results[0].cv_task
        if task_type == TaskType.SEMANTIC_SEGMENTATION:
            get_category_names = lambda eval_result: set(eval_result.mp.class_names)
        else:
            get_category_names = lambda eval_result: set(eval_result.mp.cat_names)

        category_names = get_category_names(self.eval_results[0])
        max_categories = len(category_names)
        for eval_result in self.eval_results[1:]:
            current_category_names = get_category_names(eval_result)
            category_names.intersection_update(current_category_names)
            max_categories = max(max_categories, len(current_category_names))

        if not category_names:
            raise RuntimeError(
                "Comparison validation failed: No overlapping categories found in the evaluations."
            )

        if len(category_names) == max_categories:
            logger.info("All categories are identical in all evaluations, no filtering applied.")
            return

        self.matched_classes_dict = {
            "max": max_categories,
            "current": len(category_names),
            "percentage": f"{len(category_names) / max_categories * 100:.2f}%",
        }
        self.classes_partially_matched = True
        logger.info(
            "Found %s overlapping categories across evaluations",
            f"{self.matched_classes_dict['current']}/{self.matched_classes_dict['max']} ({self.matched_classes_dict['percentage']})",
        )

        if task_type == TaskType.SEMANTIC_SEGMENTATION:
            for eval_result in self.eval_results:
                eval_result.mp.class_names = list(
                    filter(lambda c: c in category_names, eval_result.mp.class_names)
                )
                eval_result.mp.confusion_matrix = (
                    eval_result.mp.confusion_matrix[0],
                    [c for c in eval_result.mp.confusion_matrix[1] if c in category_names],
                )
        else:
            for eval_result in self.eval_results:
                eval_result.mp.cat_names = list(
                    filter(lambda c: c in category_names, eval_result.mp.cat_names)
                )
                eval_result.click_data.objects_by_class = dict(
                    filter(
                        lambda item: item[0] in category_names,
                        eval_result.click_data.objects_by_class.items(),
                    )
                )
                eval_result.click_data.outcome_counts_by_class = dict(
                    filter(
                        lambda item: item[0] in category_names,
                        eval_result.click_data.outcome_counts_by_class.items(),
                    )
                )

        for eval_result in self.eval_results:
            eval_result.inference_info["inference_settings"]["classes"] = list(category_names)
            if task_type == TaskType.SEMANTIC_SEGMENTATION:
                bar_data, labels = eval_result.mp.classwise_segm_error_data
                names = list(labels)
                keep_idx = [i for i, n in enumerate(names) if n in category_names]

                if hasattr(bar_data, "iloc"):
                    filtered_bar_data = bar_data.iloc[keep_idx]
                else:
                    filtered_bar_data = [bar_data[i] for i in keep_idx]

                filtered_labels = [names[i] for i in keep_idx]
                eval_result.mp.classwise_segm_error_data = (filtered_bar_data, filtered_labels)

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
