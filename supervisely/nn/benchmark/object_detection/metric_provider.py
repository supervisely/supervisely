import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from supervisely._utils import logger
from supervisely.nn.benchmark.utils.detection import metrics

METRIC_NAMES = {
    "mAP": "mAP",
    "f1": "F1-score",
    "precision": "Precision",
    "recall": "Recall",
    "iou": "Avg. IoU",
    "classification_accuracy": "Classification Accuracy",
    "calibration_score": "Calibration Score",
}


def _get_outcomes_per_image(matches, cocoGt):
    """
    type cocoGt: COCO
    """
    img_ids = sorted(cocoGt.getImgIds())
    imgId2idx = {img_id: idx for idx, img_id in enumerate(img_ids)}
    outcomes_per_image = np.zeros((len(img_ids), 3), dtype=float)
    for m in matches:
        img_id = m["image_id"]
        idx = imgId2idx[img_id]
        if m["type"] == "TP":
            outcomes_per_image[idx, 0] += 1
        elif m["type"] == "FP":
            outcomes_per_image[idx, 1] += 1
        elif m["type"] == "FN":
            outcomes_per_image[idx, 2] += 1
    return img_ids, outcomes_per_image


def filter_by_conf(matches: list, conf: float):
    matches_filtered = []
    for m in matches:
        if m["score"] is not None and m["score"] < conf:
            if m["type"] == "TP":
                # TP becomes FN
                m = deepcopy(m)
                m["type"] = "FN"
                m["score"] = None
                m["dt_id"] = None
                m["iou"] = None
            elif m["type"] == "FP":
                continue
            else:
                raise ValueError("Invalid match type")
        matches_filtered.append(m)
    return matches_filtered


class MetricProvider:
    def __init__(self, eval_data: dict, cocoGt, cocoDt):
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
        self.eval_data = eval_data
        self.matches = eval_data["matches"]
        self.coco_metrics = eval_data["coco_metrics"]
        self.params = eval_data["params"]
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt
        self.coco_mAP = self.coco_metrics["mAP"]
        self.coco_precision = self.coco_metrics["precision"]
        self.iouThrs = self.params["iouThrs"]
        self.recThrs = self.params["recThrs"]

        self.metric_names = METRIC_NAMES

        # metainfo
        self.cat_ids = cocoGt.getCatIds()
        self.cat_names = [cocoGt.cats[cat_id]["name"] for cat_id in self.cat_ids]

        # Evaluation params
        eval_params = self.params.get("evaluation_params", {})
        self.iou_threshold = eval_params.get("iou_threshold", 0.5)
        self.iou_threshold_idx = np.where(np.isclose(self.iouThrs, self.iou_threshold))[0][0]
        self.iou_threshold_per_class = eval_params.get("iou_threshold_per_class")
        self.iou_idx_per_class = self.params.get("iou_idx_per_class")  # {cat id: iou_idx}
        self.average_across_iou_thresholds = eval_params.get("average_across_iou_thresholds", True)

    def calculate(self):
        self.m_full = _MetricProvider(self.matches, self.eval_data, self.cocoGt, self.cocoDt)
        self.m_full._calculate_score_profile()

        # Find optimal confidence threshold
        self.f1_optimal_conf, self.best_f1 = self.m_full.get_f1_optimal_conf()
        self.custom_conf_threshold, self.custom_f1 = self.m_full.get_custom_conf_threshold()

        # Confidence threshold that will be used in visualizations
        self.conf_threshold = self.custom_conf_threshold or self.f1_optimal_conf
        if self.conf_threshold is None:
            raise RuntimeError("Model predicted no TP matches. Cannot calculate metrics.")

        # Filter by optimal confidence threshold
        if self.conf_threshold is not None:
            matches_filtered = filter_by_conf(self.matches, self.conf_threshold)
        else:
            matches_filtered = self.matches
        self.m = _MetricProvider(matches_filtered, self.eval_data, self.cocoGt, self.cocoDt)
        self.matches_filtered = matches_filtered
        self.m._init_counts()

        self.ious = self.m.ious
        self.TP_count = self.m.TP_count
        self.FP_count = self.m.FP_count
        self.FN_count = self.m.FN_count
        self.true_positives = self.m.true_positives
        self.false_negatives = self.m.false_negatives
        self.false_positives = self.m.false_positives
        self.confused_matches = self.m.confused_matches

        self.score_profile_f1s = self.m_full.score_profile_f1s

        # base metrics
        self._base_metrics = self.m.base_metrics()
        self._per_class_metrics = self.m.per_class_metrics()
        self._pr_curve = self.m.pr_curve()
        self._prediction_table = self.m.prediction_table()
        self._confusion_matrix = self.m.confusion_matrix()
        self._frequently_confused = self.m.frequently_confused(self._confusion_matrix)
        # calibration metrics
        self._confidence_score_profile = self.m_full.confidence_score_profile()
        self._calibration_curve = self.m_full.calibration_curve()
        self._scores_tp_and_fp = self.m_full.scores_tp_and_fp()
        self._maximum_calibration_error = self.m_full.maximum_calibration_error()
        self._expected_calibration_error = self.m_full.expected_calibration_error()

    def json_metrics(self):
        base = self.base_metrics()
        iou_name = int(self.iou_threshold * 100)
        if self.iou_threshold_per_class is not None:
            iou_name = "_custom"
        ap_by_class = self.AP_per_class().tolist()
        ap_by_class = dict(zip(self.cat_names, ap_by_class))
        ap_custom_by_class = self.AP_custom_per_class().tolist()
        ap_custom_by_class = dict(zip(self.cat_names, ap_custom_by_class))
        data = {
            "mAP": base["mAP"],
            "AP50": self.coco_metrics.get("AP50"),
            "AP75": self.coco_metrics.get("AP75"),
            f"AP{iou_name}": self.AP_custom(),
            "f1": base["f1"],
            "precision": base["precision"],
            "recall": base["recall"],
            "iou": base["iou"],
            "classification_accuracy": base["classification_accuracy"],
            "calibration_score": base["calibration_score"],
            "f1_optimal_conf": self.f1_optimal_conf,
            "expected_calibration_error": self.expected_calibration_error(),
            "maximum_calibration_error": self.maximum_calibration_error(),
            "AP_by_class": ap_by_class,
            f"AP{iou_name}_by_class": ap_custom_by_class,
        }
        if self.custom_conf_threshold is not None:
            data["custom_confidence_threshold"] = self.custom_conf_threshold
        return data

    def key_metrics(self):
        iou_name = int(self.iou_threshold * 100)
        if self.iou_threshold_per_class is not None:
            iou_name = "_custom"
        json_metrics = self.json_metrics()
        json_metrics.pop("AP_by_class")
        json_metrics.pop(f"AP{iou_name}_by_class")
        return json_metrics

    def metric_table(self):
        table = self.json_metrics()
        iou_name = int(self.iou_threshold * 100)
        if self.iou_threshold_per_class is not None:
            iou_name = "_custom"
        data = {
            "mAP": table["mAP"],
            "AP50": table["AP50"],
            "AP75": table["AP75"],
            f"AP{iou_name}": table[f"AP{iou_name}"],
            "f1": table["f1"],
            "precision": table["precision"],
            "recall": table["recall"],
            "Avg. IoU": table["iou"],
            "Classification Acc.": table["classification_accuracy"],
            "Calibration Score": table["calibration_score"],
            "Optimal confidence threshold": table["f1_optimal_conf"],
        }
        if self.custom_conf_threshold is not None:
            data["Custom confidence threshold"] = table["custom_confidence_threshold"]
        return data

    def AP_per_class(self):
        s = self.coco_precision[:, :, :, 0, 2].copy()
        s[s == -1] = np.nan
        ap = np.nanmean(s, axis=(0, 1))
        ap = np.nan_to_num(ap, nan=0)
        return ap

    def AP_custom_per_class(self):
        s = self.coco_precision[self.iou_threshold_idx, :, :, 0, 2]
        s = s.copy()
        if self.iou_threshold_per_class is not None:
            for cat_id, iou_idx in self.iou_idx_per_class.items():
                s[:, cat_id - 1] = self.coco_precision[iou_idx, :, cat_id - 1, 0, 2]
        s[s == -1] = np.nan
        ap = np.nanmean(s, axis=0)
        ap = np.nan_to_num(ap, nan=0)
        return ap

    def AP_custom(self):
        return np.nanmean(self.AP_custom_per_class())

    def base_metrics(self):
        base = self._base_metrics
        calibration_score = 1 - self._expected_calibration_error
        return {**base, "calibration_score": calibration_score}

    def per_class_metrics(self):
        return self._per_class_metrics

    def pr_curve(self):
        return self._pr_curve

    def prediction_table(self):
        return self._prediction_table

    def confusion_matrix(self):
        return self._confusion_matrix

    def frequently_confused(self):
        return self._frequently_confused

    def confidence_score_profile(self):
        return self._confidence_score_profile

    def calibration_curve(self):
        return self._calibration_curve

    def scores_tp_and_fp(self):
        return self._scores_tp_and_fp

    def maximum_calibration_error(self):
        return self._maximum_calibration_error

    def expected_calibration_error(self):
        return self._expected_calibration_error

    def get_f1_optimal_conf(self):
        return self.f1_optimal_conf, self.best_f1


class _MetricProvider:
    def __init__(self, matches: list, eval_data: dict, cocoGt, cocoDt):
        """
        type cocoGt: COCO
        type cocoDt: COCO
        """

        self.matches = matches
        self.eval_data = eval_data
        self.coco_metrics = eval_data["coco_metrics"]
        self.params = eval_data["params"]
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt
        self.coco_mAP = self.coco_metrics["mAP"]
        self.coco_precision = self.coco_metrics["precision"]
        self.iouThrs = self.params["iouThrs"]
        self.recThrs = self.params["recThrs"]

        # metainfo
        self.cat_ids = cocoGt.getCatIds()
        self.cat_names = [cocoGt.cats[cat_id]["name"] for cat_id in self.cat_ids]

        # Matches
        self.tp_matches = [m for m in self.matches if m["type"] == "TP"]
        self.fp_matches = [m for m in self.matches if m["type"] == "FP"]
        self.fn_matches = [m for m in self.matches if m["type"] == "FN"]
        self.confused_matches = [m for m in self.fp_matches if m["miss_cls"]]
        self.fp_not_confused_matches = [m for m in self.fp_matches if not m["miss_cls"]]
        self.ious = np.array([m["iou"] for m in self.tp_matches])

        # Evaluation params
        self.iou_idx_per_class = np.array(
            [self.params["iou_idx_per_class"][cat_id] for cat_id in self.cat_ids]
        )[:, None]
        eval_params = self.params.get("evaluation_params", {})
        self.average_across_iou_thresholds = eval_params.get("average_across_iou_thresholds", True)

    def _init_counts(self):
        cat_ids = self.cat_ids
        iouThrs = self.iouThrs
        cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        ious = []
        cats = []
        for match in self.tp_matches:
            ious.append(match["iou"])
            cats.append(cat_id_to_idx[match["category_id"]])
        ious = np.array(ious) + np.spacing(1)
        iou_idxs = np.searchsorted(iouThrs, ious) - 1
        cats = np.array(cats)
        # TP
        true_positives = np.histogram2d(
            cats,
            iou_idxs,
            bins=(len(cat_ids), len(iouThrs)),
            range=((0, len(cat_ids)), (0, len(iouThrs))),
        )[0].astype(int)
        true_positives = true_positives[:, ::-1].cumsum(1)[:, ::-1]
        tp_count = true_positives[:, 0]
        # FN
        cats_fn = np.array([cat_id_to_idx[match["category_id"]] for match in self.fn_matches])
        if cats_fn.size == 0:
            fn_count = np.zeros((len(cat_ids),), dtype=int)
        else:
            fn_count = np.bincount(cats_fn, minlength=len(cat_ids)).astype(int)
        gt_count = fn_count + tp_count
        false_negatives = gt_count[:, None] - true_positives
        # FP
        cats_fp = np.array([cat_id_to_idx[match["category_id"]] for match in self.fp_matches])
        if cats_fp.size == 0:
            fp_count = np.zeros((len(cat_ids),), dtype=int)
        else:
            fp_count = np.bincount(cats_fp, minlength=len(cat_ids)).astype(int)
        dt_count = fp_count + tp_count
        false_positives = dt_count[:, None] - true_positives

        self.true_positives = true_positives
        self.false_negatives = false_negatives
        self.false_positives = false_positives
        self.TP_count = int(self._take_iou_thresholds(true_positives).sum())
        self.FP_count = int(self._take_iou_thresholds(false_positives).sum())
        self.FN_count = int(self._take_iou_thresholds(false_negatives).sum())

        # self.true_positives = self.eval_data["true_positives"]
        # self.false_negatives = self.eval_data["false_negatives"]
        # self.false_positives = self.eval_data["false_positives"]
        # self.TP_count = int(self._take_iou_thresholds(self.true_positives).sum())
        # self.FP_count = int(self._take_iou_thresholds(self.false_positives).sum())
        # self.FN_count = int(self._take_iou_thresholds(self.false_negatives).sum())

    def _take_iou_thresholds(self, x):
        return np.take_along_axis(x, self.iou_idx_per_class, axis=1)

    def base_metrics(self):
        if self.average_across_iou_thresholds:
            tp = self.true_positives
            fp = self.false_positives
            fn = self.false_negatives
        else:
            tp = self._take_iou_thresholds(self.true_positives)
            fp = self._take_iou_thresholds(self.false_positives)
            fn = self._take_iou_thresholds(self.false_negatives)
        confuse_count = len(self.confused_matches)

        mAP = self.coco_mAP
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1 = 2 * precision * recall / (precision + recall)
        f1[(precision + recall) == 0.0] = 0.0
        iou = np.mean(self.ious)
        classification_accuracy = self.TP_count / (self.TP_count + confuse_count)

        return {
            "mAP": mAP,
            "f1": np.nanmean(f1),
            "precision": np.nanmean(precision),
            "recall": np.nanmean(recall),
            "iou": iou,
            "classification_accuracy": classification_accuracy,
        }

    def per_class_metrics(self):
        if self.average_across_iou_thresholds:
            tp = self.true_positives.mean(1)
            fp = self.false_positives.mean(1)
            fn = self.false_negatives.mean(1)
        else:
            tp = self._take_iou_thresholds(self.true_positives).flatten()
            fp = self._take_iou_thresholds(self.false_positives).flatten()
            fn = self._take_iou_thresholds(self.false_negatives).flatten()
        pr = tp / (tp + fp)
        rc = tp / (tp + fn)
        f1 = 2 * pr * rc / (pr + rc)
        return pd.DataFrame({"category": self.cat_names, "precision": pr, "recall": rc, "f1": f1})

    def pr_curve(self):
        pr_curve = self.coco_precision[:, :, :, 0, 2].mean(0)
        return pr_curve

    def prediction_table(self):
        img_ids, outcomes_per_image = _get_outcomes_per_image(self.matches, self.cocoGt)
        sly_ids = [self.cocoGt.imgs[img_id]["sly_id"] for img_id in img_ids]
        image_names = [self.cocoGt.imgs[img_id]["file_name"] for img_id in img_ids]
        n_gt = outcomes_per_image[:, 0] + outcomes_per_image[:, 2]
        n_dt = outcomes_per_image[:, 0] + outcomes_per_image[:, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision_per_image = outcomes_per_image[:, 0] / n_dt
            recall_per_image = outcomes_per_image[:, 0] / n_gt
            f1_per_image = (
                2
                * precision_per_image
                * recall_per_image
                / (precision_per_image + recall_per_image)
            )
        prediction_table = pd.DataFrame(
            {
                "Sly ID": sly_ids,
                "Image name": image_names,
                "GT objects": n_gt,
                "Predictions": n_dt,
                "TP": outcomes_per_image[:, 0],
                "FP": outcomes_per_image[:, 1],
                "FN": outcomes_per_image[:, 2],
                "Precision": precision_per_image,
                "Recall": recall_per_image,
                "F1": f1_per_image,
            }
        )
        return prediction_table

    def confusion_matrix(self):
        K = len(self.cat_ids)
        cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        confusion_matrix = np.zeros((K + 1, K + 1), dtype=int)

        for m in self.confused_matches:
            cat_idx_pred = cat_id_to_idx[m["category_id"]]
            cat_idx_gt = cat_id_to_idx[self.cocoGt.anns[m["gt_id"]]["category_id"]]
            confusion_matrix[cat_idx_pred, cat_idx_gt] += 1

        for m in self.tp_matches:
            cat_idx = cat_id_to_idx[m["category_id"]]
            confusion_matrix[cat_idx, cat_idx] += 1

        for m in self.fp_not_confused_matches:
            cat_idx_pred = cat_id_to_idx[m["category_id"]]
            confusion_matrix[cat_idx_pred, -1] += 1

        for m in self.fn_matches:
            cat_idx_gt = cat_id_to_idx[m["category_id"]]
            confusion_matrix[-1, cat_idx_gt] += 1

        return confusion_matrix

    def frequently_confused(self, confusion_matrix, topk_pairs=20):
        # Frequently confused class pairs
        cat_id_enum = {i: cat_id for i, cat_id in enumerate(self.cat_ids)}
        cm = confusion_matrix[:-1, :-1]
        cm_l = np.tril(cm, -1)
        cm_u = np.triu(cm, 1)
        cm = cm_l + cm_u.T
        cm_flat = cm.flatten()
        inds_sort = np.argsort(-cm_flat)[:topk_pairs]
        inds_sort = inds_sort[cm_flat[inds_sort] > 0]  # remove zeros
        inds_sort = np.unravel_index(inds_sort, cm.shape)

        # probability of confusion: (predicted A, actually B + predicted B, actually A) / (predicted A + predicted B)
        confused_counts = cm[inds_sort]
        dt_total = confusion_matrix.sum(1)
        dt_pair_sum = np.array([dt_total[i] + dt_total[j] for i, j in zip(*inds_sort)])
        confused_prob = confused_counts / dt_pair_sum
        inds_sort2 = np.argsort(-confused_prob)

        confused_idxs = np.array(inds_sort).T[inds_sort2]
        confused_name_pairs = [(self.cat_names[i], self.cat_names[j]) for i, j in confused_idxs]
        confused_counts = confused_counts[inds_sort2]
        confused_prob = confused_prob[inds_sort2]
        confused_catIds = [(cat_id_enum[i], cat_id_enum[j]) for i, j in confused_idxs]

        return pd.DataFrame(
            {
                "category_pair": confused_name_pairs,
                "category_id_pair": confused_catIds,
                "count": confused_counts,
                "probability": confused_prob,
            }
        )

    def _calculate_score_profile(self):
        iouThrs = self.iouThrs
        n_gt = len(self.tp_matches) + len(self.fn_matches)
        matches_sorted = sorted(
            self.tp_matches + self.fp_matches, key=lambda x: x["score"], reverse=True
        )
        scores = np.array([m["score"] for m in matches_sorted])
        ious = np.array([m["iou"] if m["type"] == "TP" else 0.0 for m in matches_sorted])
        iou_idxs = np.searchsorted(iouThrs, ious + np.spacing(1))

        # Check
        tps = np.array([m["type"] == "TP" for m in matches_sorted])
        assert np.all(iou_idxs[tps] > 0)
        assert np.all(iou_idxs[~tps] == 0)

        f1s = []
        pr_line = np.zeros(len(scores))
        rc_line = np.zeros(len(scores))
        for iou_idx, iou_th in enumerate(iouThrs):
            tps = iou_idxs > iou_idx
            fps = ~tps
            tps_sum = np.cumsum(tps)
            fps_sum = np.cumsum(fps)
            precision = tps_sum / (tps_sum + fps_sum)
            recall = tps_sum / n_gt
            f1 = 2 * precision * recall / (precision + recall)
            pr_line = pr_line + precision
            rc_line = rc_line + recall
            f1s.append(f1)
        pr_line /= len(iouThrs)
        rc_line /= len(iouThrs)
        f1s = np.array(f1s)
        # f1_line = f1s.mean(axis=0)
        f1_line = np.nanmean(f1s, axis=0)
        self.score_profile = {
            "scores": scores,
            "precision": pr_line,
            "recall": rc_line,
            "f1": f1_line,
        }
        self.score_profile_f1s = f1s

        self.iou_idxs = iou_idxs
        self.scores = scores
        self.y_true = iou_idxs > 0

    # def confidence_score_profile_v0(self):
    #     n_gt = len(self.tp_matches) + len(self.fn_matches)
    #     matches_sorted = sorted(self.tp_matches + self.fp_matches, key=lambda x: x['score'], reverse=True)
    #     scores = np.array([m["score"] for m in matches_sorted])
    #     tps = np.array([m["type"] == "TP" for m in matches_sorted])
    #     fps = ~tps
    #     tps_sum = np.cumsum(tps)
    #     fps_sum = np.cumsum(fps)
    #     precision = tps_sum / (tps_sum + fps_sum)
    #     recall = tps_sum / n_gt
    #     f1 = 2 * precision * recall / (precision + recall)
    #     return {
    #         "scores": scores,
    #         "precision": precision,
    #         "recall": recall,
    #         "f1": f1
    #     }

    def confidence_score_profile(self):
        return self.score_profile

    def get_f1_optimal_conf(self):
        if (~np.isnan(self.score_profile["f1"])).sum() == 0:
            return None, None
        argmax = np.nanargmax(self.score_profile["f1"])
        f1_optimal_conf = self.score_profile["scores"][argmax]
        best_f1 = self.score_profile["f1"][argmax]
        return f1_optimal_conf, best_f1

    def get_custom_conf_threshold(self):
        if (~np.isnan(self.score_profile["f1"])).sum() == 0:
            return None, None
        conf_threshold = self.params.get("evaluation_params", {}).get("confidence_threshold")
        if conf_threshold is not None and conf_threshold != "auto":
            idx = np.argmin(np.abs(self.score_profile["scores"] - conf_threshold))
            custom_f1 = self.score_profile["f1"][idx]
            return conf_threshold, custom_f1
        return None, None

    def calibration_curve(self):
        from sklearn.calibration import (  # pylint: disable=import-error
            calibration_curve,
        )

        true_probs, pred_probs = calibration_curve(self.y_true, self.scores, n_bins=10)
        return true_probs, pred_probs

    def maximum_calibration_error(self):
        return metrics.maximum_calibration_error(self.y_true, self.scores, n_bins=10)

    def expected_calibration_error(self):
        return metrics.expected_calibration_error(self.y_true, self.scores, n_bins=10)

    def scores_tp_and_fp(self):
        tps = self.y_true
        scores_tp = self.scores[tps]
        scores_fp = self.scores[~tps]
        return scores_tp, scores_fp
