from typing import Any, Dict, List

import numpy as np
import pandas as pd

METRIC_NAMES = {
    "mPixel": "mPixel accuracy",
    "mF1": "mF1-score",
    "mPrecision": "mPrecision",
    "mRecall": "mRecall",
    "mIoU": "mIoU",
    "mBoundaryIoU": "mBoundaryIoU",
    "calibration_score": "Calibration Score",
}


class MetricProvider:
    def __init__(self, eval_data: Dict[str, Any]):
        """
        Main class for calculating prediction metrics.

        :param matches: dictionary with matches between ground truth and predicted objects
        :type matches: list
        :param coco_metrics: dictionary with COCO metrics
        :type coco_metrics: dict
        :param params: dictionary with evaluation parameters
        :type params: dict
        :param cocoGt: COCO object with ground truth annotations
        :type cocoGt: COCO
        :param cocoDt: COCO object with predicted annotations
        :type cocoDt: COCO
        """

        # self.params = params
        self.metric_names = METRIC_NAMES

        # eval_data
        self.eval_data = eval_data["result"]
        self.class_names = eval_data.index.tolist()  # ??

        self.num_classes = len(self.class_names)

        # base metrics
        overall_TP = self.eval_data["TP"][: self.num_classes].sum()
        overall_FN = self.eval_data["FN"][: self.num_classes].sum()
        self.pixel_accuracy = overall_TP / (overall_TP + overall_FN)
        self.overall_TP = overall_TP
        self.overall_FN = overall_FN
        self.precision = round(self.eval_data.loc["mean", "precision"] * 100, 1)
        self.recall = round(self.eval_data.loc["mean", "recall"] * 100, 1)
        self.f1_score = round(self.eval_data.loc["mean", "F1_score"] * 100, 1)
        self.iou = round(self.eval_data.loc["mean", "IoU"] * 100, 1)
        self.boundary_iou = round(self.eval_data.loc["mean", "boundary_IoU"] * 100, 1)

        # error metrics
        # labels = ["mIoU", "mBoundaryEoU", "mExtentEoU", "mSegmentEoU"]
        self.boundary_eou = round(self.eval_data.loc["mean", "E_boundary_oU"] * 100, 1)
        self.extent_eou = round(self.eval_data.loc["mean", "E_extent_oU"] * 100, 1)
        self.segment_eou = round(self.eval_data.loc["mean", "E_segment_oU"] * 100, 1)

        # renormalized error metrics
        # labels = ["boundary", "extent", "segment"]
        self.boundary_renormed_eou = round(
            self.eval_data.loc["mean", "E_boundary_oU_renormed"] * 100, 1
        )
        self.extent_renormed_eou = round(
            self.eval_data.loc["mean", "E_extent_oU_renormed"] * 100, 1
        )
        self.segment_renormed_eou = round(
            self.eval_data.loc["mean", "E_segment_oU_renormed"] * 100, 1
        )

        # classwise error data
        self.classwise_segm_error_data = self.classwise_error_data()

        # confusion matrix
        self.confusion_matrix = self.confusion_matrix(self.eval_data["confusion_matrix"].copy())

        # frequently confused classes

    # def calculate(self):
    #     # base metrics
    #     self._base_metrics = self.m.base_metrics()
    #     self._confusion_matrix = self.m.confusion_matrix()
    #     self._frequently_confused = self.m.frequently_confused(self._confusion_matrix)

    def json_metrics(self):
        pass

    def metric_table(self):
        pass

    def confusion_matrix(self, confusion_matrix: np.ndarray):
        if len(self.eval_data.index) > 7:
            original_classes = self.eval_data.index.tolist()
            per_class_iou = self.eval_data["IoU"].copy()
            remove_classes = per_class_iou.index[7:].tolist()
            remove_indexes = [original_classes.index(cls) for cls in remove_classes]
            confusion_matrix = np.delete(confusion_matrix, remove_indexes, 0)
            confusion_matrix = np.delete(confusion_matrix, remove_indexes, 1)
            class_names = [cls for cls in class_names if cls not in remove_classes]
            # title_text = "Confusion matrix<br><sup>(7 classes with highest error rates)</sup>"
        else:
            class_names = self.eval_data.index.tolist()
            # title_text = "Confusion matrix"

        confusion_matrix = confusion_matrix[::-1]
        # x = class_names
        # y = x[::-1].copy()
        # text_anns = [[str(el) for el in row] for row in confusion_matrix]
        return confusion_matrix  # TODO: move to visualization ?

    def frequently_confused(self):
        return self._frequently_confused

    def basic_metrics(self):
        pass

    def error_metrics(self):
        pass

    def classwise_error_data(self):
        self.eval_data.drop(["mean"], inplace=True)
        if len(self.eval_data.index) > 7:
            per_class_iou = self.eval_data["IoU"].copy()
            per_class_iou.sort_values(ascending=True, inplace=True)
            target_classes = per_class_iou.index[:7].tolist()
            # title_text = "Classwise segmentation error analysis<br><sup>(7 classes with highest error rates)</sup>"
            # labels = target_classes[::-1]
            bar_data = self.eval_data.loc[target_classes].copy()
        else:
            # title_text = "Classwise segmentation error analysis"
            bar_data = self.eval_data.copy()
        bar_data = bar_data[["IoU", "E_extent_oU", "E_boundary_oU", "E_segment_oU"]]
        bar_data.sort_values(by="IoU", ascending=False, inplace=True)
        # if not len(self.eval_data.index) > 7:
        #     labels = list(bar_data.index)
        # color_palette = ["cornflowerblue", "moccasin", "lightgreen", "orangered"]
        return bar_data  # TODO: move to visualization ?
