from __future__ import annotations

import os
import pickle
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
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
from supervisely.sly_logger import logger


class ObjectDetectionEvalResult(BaseEvalResult):
    mp_cls = MetricProvider
    PRIMARY_METRIC = "mAP"

    def _read_files(self, path: str) -> None:
        """Read all necessary files from the directory"""

        self.coco_gt = Path(path) / "cocoGt.json"
        self.coco_dt = Path(path) / "cocoDt.json"

        if self.coco_gt.exists() and self.coco_dt.exists():
            self.coco_gt, self.coco_dt = read_coco_datasets(self.coco_gt, self.coco_dt)

        eval_data_pickle_path = Path(path) / "eval_data.pkl"
        eval_data_archive_path = Path(path) / "eval_data.zip"
        eval_data_json_path = Path(path) / "eval_data.json"
        if eval_data_pickle_path.exists():
            try:
                with open(Path(path, "eval_data.pkl"), "rb") as f:
                    self.eval_data = pickle.load(f)
            except Exception:
                logger.warning("Failed to load eval_data.pkl.")
                self.eval_data = None
        if (
            self.eval_data is None
            and eval_data_archive_path.exists()
            and eval_data_json_path.exists()
        ):
            try:
                self.eval_data = self._load_eval_data_archive(
                    eval_data_archive_path, eval_data_json_path
                )
            except Exception:
                logger.warning("Failed to load eval_data from archive.")
                self.eval_data = None

        if self.eval_data is None:
            raise ValueError("Failed to load eval_data.")

        inference_info_path = Path(path) / "inference_info.json"
        if inference_info_path.exists():
            self.inference_info = load_json_file(str(inference_info_path))

        speedtest_info_path = Path(path).parent / "speedtest" / "speedtest.json"
        if speedtest_info_path.exists():
            self.speedtest_info = load_json_file(str(speedtest_info_path))

    def _load_eval_data_archive(self, path: Path, json_path: Path) -> Dict:
        """Load eval_data from archive"""
        with zipfile.ZipFile(path, mode="r") as zf:
            data = load_json_file(str(json_path))
            return self._process_value_from_archive(data, zf)

    def _process_value_from_archive(self, value, zf: zipfile.ZipFile):
        """Recursively process values from archive, handling nested dicts and lists."""
        if isinstance(value, str) and value.endswith(".npy"):
            with zf.open(value) as arr_f:
                return np.load(arr_f)
        elif isinstance(value, str) and value.endswith(".parquet"):
            with zf.open(value) as df_f:
                return pd.read_parquet(df_f)
        elif isinstance(value, str) and value.endswith(".csv"):
            with zf.open(value) as df_f:
                return pd.read_csv(df_f, sep="\t", index_col=0)
        elif isinstance(value, dict):
            res = {}
            for k, v in value.items():
                k = int(k) if isinstance(k, str) and k.isdigit() else k
                res[k] = self._process_value_from_archive(v, zf)
            return res
        elif isinstance(value, list):
            return [self._process_value_from_archive(item, zf) for item in value]
        elif isinstance(value, str) and value.isdigit():
            return int(value)
        else:
            return value

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

    def _dump_eval_results_archive(self):
        with zipfile.ZipFile(os.path.join(self.result_dir, "eval_data.zip"), mode="w") as zf:
            data = self._process_value_for_archive(self.eval_data, "", zf)
            filepath = os.path.join(self.result_dir, "eval_data.json")
            dump_json_file(data, filepath, indent=4)
            zf.write(filepath, arcname="eval_data.json")

    def _process_value_for_archive(self, value, key_prefix: str, zf: zipfile.ZipFile):
        """Recursively process values for archiving, handling nested dicts and lists."""
        if isinstance(value, np.ndarray):
            filename = f"{key_prefix}.npy" if key_prefix else "array.npy"
            filepath = os.path.join(self.result_dir, filename)
            np.save(filepath, value)
            zf.write(filepath, arcname=filename)
            os.remove(filepath)
            return filename
        elif isinstance(value, pd.DataFrame):
            filename = f"{key_prefix}.csv" if key_prefix else "dataframe.csv"
            filepath = os.path.join(self.result_dir, filename)
            value.to_csv(filepath, sep="\t")
            zf.write(filepath, arcname=filename)
            os.remove(filepath)
            return filename
        elif isinstance(value, dict):
            return {
                k: self._process_value_for_archive(v, f"{key_prefix}.{k}" if key_prefix else k, zf)
                for k, v in value.items()
            }
        elif isinstance(value, list):
            return [
                self._process_value_for_archive(item, f"{key_prefix}[{i}]", zf)
                for i, item in enumerate(value)
            ]
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, str) and value.isdigit():
            return int(value)
        else:
            return value

    def _dump_eval_results(self):
        cocoGt_path, cocoDt_path, _ = self._get_eval_paths()
        dump_json_file(self.cocoGt_json, cocoGt_path, indent=None)
        dump_json_file(self.cocoDt_json, cocoDt_path, indent=None)
        self._dump_eval_results_archive()

    def _get_eval_paths(self):
        base_dir = self.result_dir
        cocoGt_path = os.path.join(base_dir, "cocoGt.json")
        cocoDt_path = os.path.join(base_dir, "cocoDt.json")
        eval_data_path = os.path.join(base_dir, "eval_data.pkl")
        return cocoGt_path, cocoDt_path, eval_data_path
