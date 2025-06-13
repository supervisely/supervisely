from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd

from supervisely.io.json import dump_json_file, load_json_file
from supervisely.nn.benchmark.base_evaluator import BaseEvalResult, BaseEvaluator
from supervisely.nn.benchmark.object_detection.metric_provider import MetricProvider
from supervisely.nn.benchmark.utils import (
    calculate_metrics,
    read_coco_datasets,
    sly2coco,
)
from supervisely.nn.benchmark.visualization.vis_click_data import ClickData, IdMapper


class ObjectDetectionEvalResult(BaseEvalResult):
    mp_cls = MetricProvider
    PRIMARY_METRIC = "mAP"

    def _read_files(self, path: str) -> None:
        """Read all necessary files from the directory"""

        self.coco_gt = Path(path) / "cocoGt.json"
        self.coco_dt = Path(path) / "cocoDt.json"

        if self.coco_gt.exists() and self.coco_dt.exists():
            self.coco_gt, self.coco_dt = read_coco_datasets(self.coco_gt, self.coco_dt)

        eval_data_path = Path(path) / "eval_data.pkl"
        if eval_data_path.exists():
            with open(Path(path, "eval_data.pkl"), "rb") as f:
                self.eval_data = pickle.load(f)

        inference_info_path = Path(path) / "inference_info.json"
        if inference_info_path.exists():
            self.inference_info = load_json_file(str(inference_info_path))

        speedtest_info_path = Path(path).parent / "speedtest" / "speedtest.json"
        if speedtest_info_path.exists():
            self.speedtest_info = load_json_file(str(speedtest_info_path))

    def _prepare_data(self) -> None:
        """Prepare data to allow easy access to the most important parts"""

        from pycocotools.coco import COCO  # pylint: disable=import-error

        if not hasattr(self, "coco_gt") or not hasattr(self, "coco_dt"):
            raise ValueError("GT and DT datasets are not provided")

        if not isinstance(self.coco_gt, COCO) and not isinstance(self.coco_dt, COCO):
            self.coco_gt, self.coco_dt = read_coco_datasets(self.coco_gt, self.coco_dt)

        self.mp = MetricProvider(
            self.eval_data,
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

        # Click data
        gt_id_mapper = IdMapper(self.coco_gt.dataset)
        dt_id_mapper = IdMapper(self.coco_dt.dataset)

        self.click_data = ClickData(self.mp.m, gt_id_mapper, dt_id_mapper)

    @classmethod
    def from_evaluator(cls, evaulator: ObjectDetectionEvaluator) -> ObjectDetectionEvalResult:
        """Method to customize loading of the evaluation result."""
        eval_result = cls()
        eval_result.eval_data = evaulator.eval_data
        eval_result.coco_gt = evaulator.cocoGt
        eval_result.coco_dt = evaulator.cocoDt
        eval_result._prepare_data()
        return eval_result

    @property
    def key_metrics(self):
        return self.mp.key_metrics()

    @property
    def different_iou_thresholds_per_class(self) -> bool:
        return self.mp.iou_threshold_per_class is not None


class ObjectDetectionEvaluator(BaseEvaluator):
    EVALUATION_PARAMS_YAML_PATH = f"{Path(__file__).parent}/evaluation_params.yaml"
    eval_result_cls = ObjectDetectionEvalResult
    accepted_shapes = ["rectangle"]

    def evaluate(self):
        try:
            self.cocoGt_json, self.cocoDt_json = self._convert_to_coco()
        except AssertionError as e:
            raise ValueError(
                f"{e}. Please make sure that your GT and DT projects are correct. "
                "If GT project has nested datasets and DT project was created with NN app, "
                "try to use newer version of NN app."
            )
        self.cocoGt, self.cocoDt = read_coco_datasets(self.cocoGt_json, self.cocoDt_json)
        with self.pbar(message="Evaluation: Calculating metrics", total=5) as p:
            self.eval_data = calculate_metrics(
                self.cocoGt,
                self.cocoDt,
                iouType="bbox",
                progress_cb=p.update,
                evaluation_params=self.evaluation_params,
            )
        self._dump_eval_results()

    @classmethod
    def validate_evaluation_params(cls, evaluation_params: dict) -> None:
        available_thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        iou_threshold = evaluation_params.get("iou_threshold")
        if iou_threshold is not None:
            assert iou_threshold in available_thres, (
                f"iou_threshold must be one of {available_thres}, " f"but got {iou_threshold}"
            )
        iou_threshold_per_class = evaluation_params.get("iou_threshold_per_class")
        if iou_threshold_per_class is not None:
            for class_name, iou_thres in iou_threshold_per_class.items():
                assert iou_thres in available_thres, (
                    f"class {class_name}: iou_threshold_per_class must be one of {available_thres}, "
                    f"but got {iou_thres}"
                )

    def _convert_to_coco(self):
        cocoGt_json = sly2coco(
            self.gt_project_path,
            is_dt_dataset=False,
            accepted_shapes=self.accepted_shapes,
            progress=self.pbar,
            classes_whitelist=self.classes_whitelist,
        )
        cocoDt_json = sly2coco(
            self.pred_project_path,
            is_dt_dataset=True,
            accepted_shapes=self.accepted_shapes,
            progress=self.pbar,
            classes_whitelist=self.classes_whitelist,
        )

        if len(cocoGt_json["annotations"]) == 0:
            raise ValueError("Not found any annotations in GT project")
        if len(cocoDt_json["annotations"]) == 0:
            raise ValueError(
                "Not found any predictions. "
                "Please make sure that your model produces predictions."
            )
        assert (
            cocoDt_json["categories"] == cocoGt_json["categories"]
        ), "Classes in GT and Pred projects must be the same"
        assert [f'{x["dataset"]}/{x["file_name"]}' for x in cocoDt_json["images"]] == [
            f'{x["dataset"]}/{x["file_name"]}' for x in cocoGt_json["images"]
        ], "Images in GT and DT projects are different"
        return cocoGt_json, cocoDt_json

    def _dump_eval_results(self):
        cocoGt_path, cocoDt_path, eval_data_path = self._get_eval_paths()
        dump_json_file(self.cocoGt_json, cocoGt_path, indent=None)
        dump_json_file(self.cocoDt_json, cocoDt_path, indent=None)
        self._dump_pickle(self.eval_data, eval_data_path)

    def _get_eval_paths(self):
        base_dir = self.result_dir
        cocoGt_path = os.path.join(base_dir, "cocoGt.json")
        cocoDt_path = os.path.join(base_dir, "cocoDt.json")
        eval_data_path = os.path.join(base_dir, "eval_data.pkl")
        return cocoGt_path, cocoDt_path, eval_data_path
