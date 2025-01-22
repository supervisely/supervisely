from __future__ import annotations

import os
from pathlib import Path

from supervisely.io.json import dump_json_file
from supervisely.nn.benchmark.object_detection.evaluator import (
    ObjectDetectionEvalResult,
    ObjectDetectionEvaluator,
)
from supervisely.nn.benchmark.object_detection.metric_provider import MetricProvider
from supervisely.nn.benchmark.utils import calculate_metrics, read_coco_datasets


class InstanceSegmentationEvalResult(ObjectDetectionEvalResult):
    mp_cls = MetricProvider
    PRIMARY_METRIC = "mAP"

    @classmethod
    def from_evaluator(
        cls, evaulator: InstanceSegmentationEvaluator
    ) -> InstanceSegmentationEvalResult:
        """Method to customize loading of the evaluation result."""
        eval_result = cls()
        eval_result.eval_data = evaulator.eval_data
        eval_result.coco_gt = evaulator.cocoGt
        eval_result.coco_dt = evaulator.cocoDt
        eval_result._prepare_data()
        return eval_result


class InstanceSegmentationEvaluator(ObjectDetectionEvaluator):
    EVALUATION_PARAMS_YAML_PATH = f"{Path(__file__).parent}/evaluation_params.yaml"
    eval_result_cls = InstanceSegmentationEvalResult
    accepted_shapes = ["polygon", "bitmap"]

    def evaluate(self):
        try:
            self.cocoGt_json, self.cocoDt_json = self._convert_to_coco()
        except AssertionError as e:
            raise ValueError(
                f"{e}. Please make sure that your GT and DT projects are correct. "
                "If GT project has nested datasets and DT project was created with NN app, "
                "try to use newer version of NN app."
            )

        self._dump_datasets()
        self.cocoGt, self.cocoDt = read_coco_datasets(self.cocoGt_json, self.cocoDt_json)
        with self.pbar(message="Evaluation: Calculating metrics", total=5) as p:
            self.eval_data = calculate_metrics(
                self.cocoGt,
                self.cocoDt,
                iouType="segm",
                progress_cb=p.update,
                evaluation_params=self.evaluation_params,
            )
        self._dump_eval_results()

    def _dump_eval_results(self):
        _, _, eval_data_path = self._get_eval_paths()
        self._dump_pickle(self.eval_data, eval_data_path)

    def _get_eval_paths(self):
        base_dir = self.result_dir
        cocoGt_path = os.path.join(base_dir, "cocoGt.json")
        cocoDt_path = os.path.join(base_dir, "cocoDt.json")
        eval_data_path = os.path.join(base_dir, "eval_data.pkl")
        return cocoGt_path, cocoDt_path, eval_data_path

    def _dump_datasets(self):
        cocoGt_path, cocoDt_path, _ = self._get_eval_paths()
        dump_json_file(self.cocoGt_json, cocoGt_path, indent=None)
        dump_json_file(self.cocoDt_json, cocoDt_path, indent=None)
