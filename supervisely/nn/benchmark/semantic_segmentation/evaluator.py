from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from supervisely.io.json import load_json_file
from supervisely.nn.benchmark.base_evaluator import BaseEvalResult, BaseEvaluator
from supervisely.nn.benchmark.semantic_segmentation.metric_provider import (
    MetricProvider,
)
from supervisely.nn.benchmark.utils import (
    calculate_semsegm_metrics as calculate_metrics,
)
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger


class SemanticSegmentationEvalResult(BaseEvalResult):
    mp_cls = MetricProvider
    PRIMARY_METRIC = "mIoU"

    def _read_files(self, path: str) -> None:
        """Read all necessary files from the directory"""

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

        self.mp = MetricProvider(self.eval_data)

    @classmethod
    def from_evaluator(
        cls, evaulator: SemanticSegmentationEvaluator
    ) -> SemanticSegmentationEvalResult:
        """Method to customize loading of the evaluation result."""
        eval_result = cls()
        eval_result.eval_data = evaulator.eval_data
        eval_result._prepare_data()
        return eval_result

    @property
    def key_metrics(self):
        return self.mp.json_metrics()


class SemanticSegmentationEvaluator(BaseEvaluator):
    EVALUATION_PARAMS_YAML_PATH = f"{Path(__file__).parent}/evaluation_params.yaml"
    eval_result_cls = SemanticSegmentationEvalResult

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bg_cls_name = None
        self.bg_cls_color = None

    def evaluate(self):
        self.bg_cls_name = self._get_bg_class_name()
        if self.bg_cls_name not in self.classes_whitelist:
            self.classes_whitelist.append(self.bg_cls_name)

        gt_prep_path, pred_prep_path = self.prepare_segmentation_data()

        self.eval_data = calculate_metrics(
            gt_dir=gt_prep_path,
            pred_dir=pred_prep_path,
            boundary_width=0.01,
            boundary_iou_d=0.02,
            num_workers=4,
            class_names=self.classes_whitelist,
            result_dir=self.result_dir,
            progress=self.pbar,
        )
        self.eval_data["bg_cls_name"] = self.bg_cls_name
        logger.info("Successfully calculated evaluation metrics")
        self._dump_eval_results()
        logger.info("Evaluation results are saved")

    def _get_palette(self, project_path):
        meta_path = Path(project_path) / "meta.json"
        meta = ProjectMeta.from_json(load_json_file(meta_path))

        palette = []
        for cls_name in self.classes_whitelist:
            obj_cls = meta.get_obj_class(cls_name)
            if obj_cls is None:
                palette.append((0, 0, 0))
            else:
                palette.append(obj_cls.color)

        return palette

    def _dump_eval_results(self):
        eval_data_path = self._get_eval_data_path()
        self._dump_pickle(self.eval_data, eval_data_path)  # TODO: maybe dump JSON?

    def _get_eval_data_path(self):
        base_dir = self.result_dir
        eval_data_path = os.path.join(base_dir, "eval_data.pkl")
        return eval_data_path

    def prepare_segmentation_data(self):
        src_dirs = [self.gt_project_path, self.pred_project_path]
        output_dirs = [
            Path(self.gt_project_path).parent / "preprocessed_gt",
            Path(self.pred_project_path).parent / "preprocessed_pred",
        ]

        for src_dir, output_dir in zip(src_dirs, output_dirs):
            if output_dir.exists():
                logger.info(f"Preprocessed data already exists in {output_dir} directory")
                continue

            palette = self._get_palette(src_dir)
            bg_cls_idx = self.classes_whitelist.index(self.bg_cls_name)
            try:
                bg_color = palette[bg_cls_idx]
            except IndexError:
                bg_color = (0, 0, 0)
            output_dir.mkdir(parents=True)
            temp_seg_dir = src_dir + "_temp"
            if not os.path.exists(temp_seg_dir):
                Project.to_segmentation_task(
                    src_dir,
                    temp_seg_dir,
                    target_classes=self.classes_whitelist,
                    bg_name=self.bg_cls_name,
                    bg_color=bg_color,
                )

            palette_lookup = np.zeros(256**3, dtype=np.int32)
            for idx, color in enumerate(palette, 1):
                key = (color[0] << 16) | (color[1] << 8) | color[2]
                palette_lookup[key] = idx

            temp_project = Project(temp_seg_dir, mode=OpenMode.READ)
            temp_project.total_items
            for dataset in temp_project.datasets:
                dataset: Dataset
                names = dataset.get_items_names()
                for name in names:
                    mask_path = dataset.get_seg_path(name)
                    mask = cv2.imread(mask_path)[:, :, ::-1]

                    mask_keys = (
                        (mask[:, :, 0].astype(np.int32) << 16)
                        | (mask[:, :, 1].astype(np.int32) << 8)
                        | mask[:, :, 2].astype(np.int32)
                    )
                    result = palette_lookup[mask_keys]
                    if name.count(".png") > 1:
                        name = name[:-4]
                    cv2.imwrite(os.path.join(output_dir, name), result)

            shutil.rmtree(temp_seg_dir)

        return output_dirs

    def _get_bg_class_name(self):
        possible_names = ["background", "bg", "unlabeled", "neutral", "__bg__"]
        logger.info(f"Searching for background class in projects. Possible names: {possible_names}")

        bg_cls_names = []
        for project_path in [self.gt_project_path, self.pred_project_path]:
            meta_path = Path(project_path) / "meta.json"
            meta = ProjectMeta.from_json(load_json_file(meta_path))

            for obj_cls in meta.obj_classes:
                if obj_cls.name in possible_names:
                    bg_cls_names.append(obj_cls.name)
                    break

        if len(bg_cls_names) == 0:
            raise ValueError("Background class not found in GT and Pred projects")

        if len(set(bg_cls_names)) > 1:
            raise ValueError(
                f"Founds multiple background class names in GT and Pred projects: {set(bg_cls_names)}"
            )

        return bg_cls_names[0]
