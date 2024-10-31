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
        speedtest_info_path = Path(self.directory, "speedtest.json")
        if speedtest_info_path.exists():
            self.speedtest_info = load_json_file(Path(self.directory, "speedtest.json"))

        self.mp = MetricProvider(self.eval_data)
        # self.mp.calculate()


class SemanticSegmentationEvaluator(BaseEvaluator):
    EVALUATION_PARAMS_YAML_PATH = f"{Path(__file__).parent}/evaluation_params.yaml"
    eval_result_cls = SemanticSegmentationEvalResult

    def evaluate(self):
        class_names, colors = self._get_classes_names_and_colors()

        gt_prep_path = Path(self.gt_project_path).parent / "preprocessed_gt"
        pred_prep_path = Path(self.pred_project_path).parent / "preprocessed_pred"
        self.prepare_segmentation_data(self.gt_project_path, gt_prep_path, colors)
        self.prepare_segmentation_data(self.pred_project_path, pred_prep_path, colors)

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

    def prepare_segmentation_data(self, source_project_dir, output_project_dir, palette):
        if os.path.exists(output_project_dir):
            logger.info(f"Preprocessed data already exists in {output_project_dir} directory")
            return
        else:
            os.makedirs(output_project_dir)

            ann_dir = "seg"

            temp_project_seg_dir = source_project_dir + "_temp"
            if not os.path.exists(temp_project_seg_dir):
                Project.to_segmentation_task(
                    source_project_dir,
                    temp_project_seg_dir,
                )

            datasets = os.listdir(temp_project_seg_dir)
            for dataset in datasets:
                if not os.path.isdir(os.path.join(temp_project_seg_dir, dataset)):
                    continue
                # convert masks to required format and save to general ann_dir
                mask_files = os.listdir(os.path.join(temp_project_seg_dir, dataset, ann_dir))
                for mask_file in tqdm(mask_files, desc="Preparing segmentation data..."):
                    mask = cv2.imread(
                        os.path.join(temp_project_seg_dir, dataset, ann_dir, mask_file)
                    )[:, :, ::-1]
                    result = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
                    # human masks to machine masks
                    for color_idx, color in enumerate(palette):
                        colormap = np.where(np.all(mask == color, axis=-1))
                        result[colormap] = color_idx
                    if mask_file.count(".png") > 1:
                        mask_file = mask_file[:-4]
                    cv2.imwrite(os.path.join(output_project_dir, mask_file), result)

            shutil.rmtree(temp_project_seg_dir)
