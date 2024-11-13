import os
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from supervisely.io.json import load_json_file
from supervisely.nn.benchmark.base_evaluator import BaseEvalResult, BaseEvaluator
from supervisely.nn.benchmark.evaluation.semantic_segmentation.beyond_iou.calculate_metrics import (
    calculate_metrics,
)
from supervisely.nn.benchmark.semantic_segmentation.metric_provider import (
    MetricProvider,
)
from supervisely.project.project import Project
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger


class SemanticSegmentationEvalResult(BaseEvalResult):
    mp_cls = MetricProvider

    def _read_eval_data(self):
        self.eval_data = pickle.load(open(Path(self.directory, "eval_data.pkl"), "rb"))
        self.inference_info = load_json_file(Path(self.directory, "inference_info.json"))
        speedtest_info_path = Path(self.directory).parent / "speedtest" / "speedtest.json"
        self.speedtest_info = None
        if speedtest_info_path.exists():
            self.speedtest_info = load_json_file(speedtest_info_path)

        self.mp = MetricProvider(self.eval_data)
        # self.mp.calculate()


class SemanticSegmentationEvaluator(BaseEvaluator):
    EVALUATION_PARAMS_YAML_PATH = f"{Path(__file__).parent}/evaluation_params.yaml"
    eval_result_cls = SemanticSegmentationEvalResult

    def evaluate(self):
        gt_name_to_color = self._get_classes_names_to_colors(self.gt_project_path)
        pred_name_to_color = self._get_classes_names_to_colors(self.pred_project_path)

        target_classes = [name for name in gt_name_to_color.keys() if name in pred_name_to_color]

        gt_palette = [gt_name_to_color[name] for name in target_classes]
        pred_palette = [pred_name_to_color[name] for name in target_classes]

        gt_prep_path = Path(self.gt_project_path).parent / "preprocessed_gt"
        pred_prep_path = Path(self.pred_project_path).parent / "preprocessed_pred"
        self.prepare_segmentation_data(
            self.gt_project_path, gt_prep_path, gt_palette, target_classes
        )
        self.prepare_segmentation_data(
            self.pred_project_path, pred_prep_path, pred_palette, target_classes
        )

        self.eval_data = calculate_metrics(
            gt_dir=gt_prep_path,
            pred_dir=pred_prep_path,
            boundary_width=0.01,
            boundary_iou_d=0.02,
            num_workers=0, # FIXME: set 4 for production
            class_names=target_classes,
            result_dir=self.result_dir,
        )
        logger.info("Successfully calculated evaluation metrics")
        self._dump_eval_results()
        logger.info("Evaluation results are saved")

    def _get_classes_names_to_colors(self, source_project_path):
        meta_path = Path(source_project_path) / "meta.json"
        meta = ProjectMeta.from_json(load_json_file(meta_path))

        return {obj.name: obj.color for obj in meta.obj_classes}

    def _dump_eval_results(self):
        eval_data_path = self._get_eval_data_path()
        self._dump_pickle(self.eval_data, eval_data_path)  # TODO: maybe dump JSON?

    def _get_eval_data_path(self):
        base_dir = self.result_dir
        eval_data_path = os.path.join(base_dir, "eval_data.pkl")
        return eval_data_path

    def prepare_segmentation_data(
        self, source_project_dir, output_project_dir, palette, target_classes
    ):
        if os.path.exists(output_project_dir):
            logger.info(f"Preprocessed data already exists in {output_project_dir} directory")
            return

        os.makedirs(output_project_dir)
        ann_dir = "seg"
        temp_project_seg_dir = source_project_dir + "_temp"
        if not os.path.exists(temp_project_seg_dir):
            default_bg_name = self._get_bg_class_name()
            Project.to_segmentation_task(
                source_project_dir,
                temp_project_seg_dir,
                target_classes=target_classes,
                default_bg_name=default_bg_name,
            )

        palette_lookup = np.zeros(256**3, dtype=np.int32)
        for idx, color in enumerate(palette, 1):
            key = (color[0] << 16) | (color[1] << 8) | color[2]
            palette_lookup[key] = idx
        datasets = os.listdir(temp_project_seg_dir)
        for dataset in datasets:
            if not os.path.isdir(os.path.join(temp_project_seg_dir, dataset)):
                continue
            # convert masks to required format and save to general ann_dir
            mask_files = os.listdir(os.path.join(temp_project_seg_dir, dataset, ann_dir))
            for mask_file in tqdm(mask_files, desc="Preparing segmentation data..."):
                mask = cv2.imread(os.path.join(temp_project_seg_dir, dataset, ann_dir, mask_file))[
                    :, :, ::-1
                ]
                mask_keys = (
                    (mask[:, :, 0].astype(np.int32) << 16)
                    | (mask[:, :, 1].astype(np.int32) << 8)
                    | mask[:, :, 2].astype(np.int32)
                )
                result = palette_lookup[mask_keys]
                if mask_file.count(".png") > 1:
                    mask_file = mask_file[:-4]
                cv2.imwrite(os.path.join(output_project_dir, mask_file), result)

        shutil.rmtree(temp_project_seg_dir)

    def _get_bg_class_name(self):
        possible_bg_names = ["background", "bg", "unlabeled", "neutral", "__bg__"]

        meta_path = Path(self.gt_project_path) / "meta.json"
        meta = ProjectMeta.from_json(load_json_file(meta_path))

        for i, obj_cls in enumerate(meta.obj_classes):
            if obj_cls.name in possible_bg_names:
                logger.info(f"Found background class: {obj_cls.name}")
                return obj_cls.name
        return "__bg__"
