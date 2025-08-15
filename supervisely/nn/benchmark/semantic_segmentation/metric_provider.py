from typing import Any, Dict, Optional

import numpy as np

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
        # self.params = params
        self.metric_names = METRIC_NAMES

        # eval_data
        self.eval_data = eval_data["result"]
        self.bg_cls_name = eval_data["bg_cls_name"]
        self.per_image_metrics = eval_data["per_image_metrics"]
        self.cmat_cell_img_names = eval_data["cell_img_names"]
        self.class_names = self.eval_data.index.tolist()

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
        self.classwise_segm_error_data = self.get_classwise_error_data()

        # confusion matrix
        self.confusion_matrix = self.get_confusion_matrix(eval_data["confusion_matrix"].copy())

        # frequently confused classes
        self.frequently_confused = self.get_frequently_confused(
            eval_data["confusion_matrix"].copy(), n_pairs=20
        )

    def json_metrics(self):
        return {
            "mIoU": self.eval_data.loc["mean"]["IoU"],
            "mE_boundary_oU": self.eval_data.loc["mean"]["E_boundary_oU"],
            "mFP_boundary_oU": self.eval_data.loc["mean"]["FP_boundary_oU"],
            "mFN_boundary_oU": self.eval_data.loc["mean"]["FN_boundary_oU"],
            "mE_boundary_oU_renormed": self.eval_data.loc["mean"]["E_boundary_oU_renormed"],
            "mE_extent_oU": self.eval_data.loc["mean"]["E_extent_oU"],
            "mFP_extent_oU": self.eval_data.loc["mean"]["FP_extent_oU"],
            "mFN_extent_oU": self.eval_data.loc["mean"]["FN_extent_oU"],
            "mE_extent_oU_renormed": self.eval_data.loc["mean"]["E_extent_oU_renormed"],
            "mE_segment_oU": self.eval_data.loc["mean"]["E_segment_oU"],
            "mFP_segment_oU": self.eval_data.loc["mean"]["FP_segment_oU"],
            "mFN_segment_oU": self.eval_data.loc["mean"]["FN_segment_oU"],
            "mE_segment_oU_renormed": self.eval_data.loc["mean"]["E_segment_oU_renormed"],
            "mPrecision": self.eval_data.loc["mean"]["precision"],
            "mRecall": self.eval_data.loc["mean"]["recall"],
            "mF1_score": self.eval_data.loc["mean"]["F1_score"],
            "PixelAcc": self.pixel_accuracy,
            "mBoundaryIoU": self.eval_data.loc["mean"]["boundary_IoU"],
        }

    def metric_table(self):
        names_map = {
            "img_names": "Image name",
            "pixel_acc": "Pixel accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1 score",
            "iou": "IoU",
            "boundary_iou": "Boundary IoU",
            "boundary_eou": "Boundary EoU",
            "extent_eou": "Extent EoU",
            "segment_eou": "Segment EoU",
            "boundary_eou_renormed": "Boundary EoU renormed",
            "extent_eou_renormed": "Extent EoU renormed",
            "segment_eou_renormed": "Segment EoU renormed",
        }
        prediction_table = self.per_image_metrics.rename(columns=names_map)
        return prediction_table

    def get_confusion_matrix(self, confusion_matrix: np.ndarray):
        class_names = self.eval_data.index.tolist()
        confusion_matrix = confusion_matrix[::-1]
        return confusion_matrix, class_names

    def get_frequently_confused(self, confusion_matrix: np.ndarray, n_pairs: Optional[int] = None):

        non_diagonal_ids = {}
        for i, idx in enumerate(np.ndindex(confusion_matrix.shape)):
            if idx[0] != idx[1]:
                non_diagonal_ids[i] = idx

        indexes_1d = np.argsort(confusion_matrix, axis=None)
        indexes_2d = [non_diagonal_ids[idx] for idx in indexes_1d if idx in non_diagonal_ids]
        if n_pairs is not None:
            indexes_2d = indexes_2d[:n_pairs]
        indexes_2d = np.asarray(indexes_2d[::-1])

        rows = indexes_2d[:, 0]
        cols = indexes_2d[:, 1]
        probs = confusion_matrix[rows, cols]
        return probs, indexes_2d

    def key_metrics(self):
        return {
            "mPixel accuracy": self.pixel_accuracy,
            "mPrecision": self.precision,
            "mRecall": self.recall,
            "mF1-score": self.f1_score,
            "mIoU": self.iou,
            "mBoundaryIoU": self.boundary_iou,
            "mPixel accuracy": self.pixel_accuracy,
        }

    def get_classwise_error_data(self):
        bar_data = self.eval_data.copy()
        bar_data.drop(["mean"], inplace=True)
        bar_data = bar_data[["IoU", "E_extent_oU", "E_boundary_oU", "E_segment_oU"]]
        bar_data.sort_values(by="IoU", ascending=False, inplace=True)
        labels = list(bar_data.index)
        return bar_data, labels
