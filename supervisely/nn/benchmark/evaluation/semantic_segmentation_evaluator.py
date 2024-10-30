import os
from pathlib import Path
from typing import List, Optional

from supervisely.app.widgets import SlyTqdm
from supervisely.io.json import load_json_file
from supervisely.nn.benchmark.evaluation import BaseEvaluator
from supervisely.nn.benchmark.evaluation.semantic_segmentation.beyond_iou.calculate_metrics import (
    calculate_metrics
)
from supervisely.nn.benchmark.evaluation.semantic_segmentation.beyond_iou.functions import (
    prepare_segmentation_data,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger


class SemanticSegmentationEvaluator(BaseEvaluator):
    EVALUATION_PARAMS_YAML_PATH = (
        f"{Path(__file__).parent}/semantic_segmentation/evaluation_params.yaml"
    )

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
        super().__init__(
            gt_project_path,
            pred_project_path,
            result_dir,
            progress,
            items_count,
            classes_whitelist,
            evaluation_params,
        )

    def evaluate(self):
        class_names, colors = self._get_classes_names_and_colors()

        gt_prep_path = Path(self.gt_project_path).parent / "preprocessed_gt"
        pred_prep_path = Path(self.dt_project_path).parent / "preprocessed_pred"
        prepare_segmentation_data(self.gt_project_path, gt_prep_path, colors)
        prepare_segmentation_data(self.dt_project_path, pred_prep_path, colors)

        self.eval_data = calculate_metrics(
            gt_dir=gt_prep_path,
            pred_dir=pred_prep_path,
            boundary_width=0.01,
            boundary_iou_d=0.02,
            num_workers=0,  # TODO: 0 for local tests, change to 4 for production
            class_names=class_names,
            result_dir=self.result_dir,
        )
        logger.info("Successfully calculated evaluation metrics")
        self._dump_eval_results()
        logger.info("Evaluation results are saved")

    def _get_classes_names_and_colors(self):
        meta_path = Path(self.gt_project_path) / "meta.json"
        meta = ProjectMeta.from_json(load_json_file(meta_path))

        class_names = [obj.name for obj in meta.obj_classes]
        colors = [obj.color for obj in meta.obj_classes]
        return class_names, colors

    def _dump_eval_results(self):
        eval_data_path = self._get_eval_data_path()
        self._dump_pickle(self.eval_data, eval_data_path) # TODO: maybe dump JSON?

    def _get_eval_data_path(self):
        base_dir = self.result_dir
        eval_data_path = os.path.join(base_dir, "eval_data.pkl")
        return eval_data_path
