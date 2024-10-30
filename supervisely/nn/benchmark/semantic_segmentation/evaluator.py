import os
import pickle
from pathlib import Path

from supervisely.io.json import dump_json_file, load_json_file
from supervisely.nn.benchmark.base_evaluator import BaseEvalResult, BaseEvaluator
from supervisely.nn.benchmark.coco_utils import read_coco_datasets, sly2coco
from supervisely.nn.benchmark.evaluation.coco import calculate_metrics
from supervisely.nn.benchmark.semantic_segmentation.metric_provider import MetricProvider
from supervisely.sly_logger import logger


class SemanticSegmentationEvalResult(BaseEvalResult):
    mp_cls = MetricProvider

    def _read_eval_data(self):
        self.eval_data = pickle.load(
            open(Path(self.local_dir, "evaluation", "eval_data.pkl"), "rb")
        )
        self.inference_info = load_json_file(
            Path(self.local_dir, "evaluation", "inference_info.json")
        )
        speedtest_info_path = Path(self.local_dir, "speedtest", "speedtest.json")
        if speedtest_info_path.exists():
            self.speedtest_info = load_json_file(
                Path(self.local_dir, "speedtest", "speedtest.json")
            )

        self.mp = MetricProvider(
            self.eval_data["matches"],
            self.eval_data["coco_metrics"],
            self.eval_data["params"],
            self.coco_gt,
            self.coco_dt,
        )
        self.mp.calculate()


class SemanticSegmentationEvaluator(BaseEvaluator):
    EVALUATION_PARAMS_YAML_PATH = f"{Path(__file__).parent}/coco/evaluation_params.yaml"
    eval_result_cls = SemanticSegmentationEvalResult

    def evaluate(self):
        class_names, colors = self._get_classes_names_and_colors()

        gt_prep_path = Path(self.gt_project_path).parent / "preprocessed_gt"
        pred_prep_path = Path(self.dt_project_path).parent / "preprocessed_pred"
        # prepare_segmentation_data(self.gt_project_path, gt_prep_path, colors) # TODO: import 
        # prepare_segmentation_data(self.dt_project_path, pred_prep_path, colors)

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
        self._dump_pickle(self.eval_data, eval_data_path)  # TODO: maybe dump JSON?

    def _get_eval_data_path(self):
        base_dir = self.result_dir
        eval_data_path = os.path.join(base_dir, "eval_data.pkl")
        return eval_data_path
