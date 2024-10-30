import json
import os
from typing import Dict, Iterable, List, Optional, Union

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

from supervisely.nn.benchmark.evaluation.semantic_segmentation.beyond_iou.metric_provider import (
    SemSegmMetricProvider,
)
from supervisely.nn.benchmark.evaluation.semantic_segmentation.beyond_iou.utils import (
    dilate_mask,
    get_contiguous_segments,
    get_exterior_boundary,
    get_interior_boundary,
    one_hot,
)

ERROR_CODES = {
    "ignore": -1,
    "unassigned": 0,
    "TP": 1,
    "TN": 2,
    "FP_boundary": 3,
    "FN_boundary": 4,
    "FP_extent": 5,
    "FN_extent": 6,
    "FP_segment": 7,
    "FN_segment": 8,
}


ERROR_PALETTE = {
    -1: (100, 100, 100),
    0: (150, 150, 150),
    1: (255, 255, 255),
    2: (0, 0, 0),
    3: (255, 200, 150),
    4: (150, 200, 255),
    5: (255, 100, 150),
    6: (150, 100, 255),
    7: (255, 0, 0),
    8: (0, 0, 255),
}


class Evaluator:
    def __init__(
        self,
        class_names: List[str],
        ignore_index: Optional[int] = None,
        boundary_width: Union[float, int] = 0.01,
        boundary_iou_d: float = 0.02,
        boundary_implementation: str = "exact",
        result_dir: str = "output",
    ):
        """The main class for running our error analysis.
        :param class_names: List of strings providing names for class ids 0,...,C.
        :param ignore_index: Class id to be ignored in the IoU computation.
        :param boundary_width: The parameter d in the paper, either as a float in [0,1] (relative to diagonal)
            or as an integer > 1 (absolute number of pixels).
        :param boundary_implementation: Choose "exact" for the euclidean pixel distance.
            The Boundary IoU paper uses the L1 distance ("fast").
        """
        global torch, np, GPU
        import torch

        if torch.cuda.is_available():
            GPU = True
            try:
                import cupy as np  # gpu-compatible numpy analogue

                global numpy
                import numpy as numpy
            except:
                import warnings

                warnings.warn(
                    "Failed to import cupy. Use cupy official documentation to install this "
                    "module: https://docs.cupy.dev/en/stable/install.html"
                )
        else:
            GPU = False
            import numpy as np

            global numpy
            numpy = np

        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.ignore_index = ignore_index

        self.boundary_width = boundary_width
        if 0 < self.boundary_width < 1:
            self.use_relative_boundary_width = True
        elif self.boundary_width % 1 != 0 or self.boundary_width < 0:
            raise ValueError("boundary_width should be an integer or a float in (0,1)!")
        else:
            self.use_relative_boundary_width = False

        self.boundary_implementation = boundary_implementation
        self.boundary_iou_d = boundary_iou_d

        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes),
        )
        self.cell_img_names = {}
        self.result_dir = result_dir

        self.image_metrics = {
            "pixel_acc": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "iou": [],
            "boundary_iou": [],
            "boundary_eou": [],
            "extent_eou": [],
            "segment_eou": [],
            "boundary_eou_renormed": [],
            "extent_eou_renormed": [],
            "segment_eou_renormed": [],
        }
        self.img_names = []

    def extract_masks(self, seg, cl, n_cl):
        if GPU:
            seg = np.asarray(seg)
        h, w = seg.shape
        masks = np.zeros((n_cl, h, w))

        for i, c in enumerate(cl):
            masks[i, :, :] = seg == c

        return masks

    def calc_confusion_matrix(self, pred, gt, cmat, img_name):
        assert pred.shape == gt.shape

        cl = np.arange(cmat.shape[0])
        n_cl = len(cl)
        pred_mask = self.extract_masks(pred, cl, n_cl)
        gt_mask = self.extract_masks(gt, cl, n_cl)

        for ig in range(n_cl):
            gm = gt_mask[ig, :, :]
            if np.sum(gm) == 0:
                continue

            for ip in range(n_cl):
                pm = pred_mask[ip, :, :]
                if np.sum(pm) == 0:
                    continue

                cmat[ig, ip] += np.sum(np.logical_and(pm, gm))
                cell = str(ig) + str(ip)
                if cell in self.cell_img_names:
                    self.cell_img_names[cell].append(img_name)
                else:
                    self.cell_img_names[cell] = [img_name]

        return cmat

    def evaluate(self, loader: Iterable):
        """This runs the analysis for a whole dataset.
        :param loader: Iterable providing pairs of (pred, gt).
        :returns: beyond_iou.Result.
        """
        self.results = {
            "unassigned": np.zeros(self.num_classes, dtype=np.int64),
            "ignore": np.zeros(self.num_classes, dtype=np.int64),
            "TP": np.zeros(self.num_classes, dtype=np.int64),
            "TN": np.zeros(self.num_classes, dtype=np.int64),
            "FP_boundary": np.zeros(self.num_classes, dtype=np.int64),
            "FN_boundary": np.zeros(self.num_classes, dtype=np.int64),
            "FP_extent": np.zeros(self.num_classes, dtype=np.int64),
            "FN_extent": np.zeros(self.num_classes, dtype=np.int64),
            "FP_segment": np.zeros(self.num_classes, dtype=np.int64),
            "FN_segment": np.zeros(self.num_classes, dtype=np.int64),
        }
        self.boundary_iou_intersection_counts = np.zeros(self.num_classes, dtype=np.int64)
        self.boundary_iou_union_counts = np.zeros(self.num_classes, dtype=np.int64)

        for pred, gt, img_name in tqdm(
            loader, total=len(loader), smoothing=0, desc="Calculating metrics..."
        ):
            sample_results = self.evaluate_sample(pred, gt, img_name)
            self.update_results(sample_results, img_name)
            self.confusion_matrix = self.calc_confusion_matrix(
                pred,
                gt,
                self.confusion_matrix,
                img_name,
            )

        if GPU:
            for key, value in self.results.items():
                self.results[key] = value.get()
            self.boundary_iou_intersection_counts = self.boundary_iou_intersection_counts.get()
            self.boundary_iou_union_counts = self.boundary_iou_union_counts.get()

        result = self.calculate_error_metrics()
        normalized_confusion_matrix = self.confusion_matrix / self.confusion_matrix.sum(
            axis=1, keepdims=True
        )
        normalized_confusion_matrix = np.round(normalized_confusion_matrix, 3)
        return {
            "result": result,
            "confusion_matrix": self.confusion_matrix,
        }

        # with open(f"{self.result_dir}/cell_img_names.json", "w") as file:
        #     json.dump(self.cell_img_names, file)

        normalized_confusion_matrix = self.confusion_matrix / self.confusion_matrix.sum(
            axis=1, keepdims=True
        )
        normalized_confusion_matrix = np.round(normalized_confusion_matrix, 3)
        # np.save(f"{self.result_dir}/confusion_matrix.npy", normalized_confusion_matrix)

        image_metrics_df = pd.DataFrame(data=self.image_metrics, index=self.img_names)
        # image_metrics_df.to_csv(f"{self.result_dir}/per_image_metrics.csv", index=True)

        # final_result = {
        #     "result": result.dataframe.to_dict(),
        #     "image_metrics": image_metrics_df.to_dict(),
        #     "confusion_matrix": self.confusion_matrix.tolist(),
        #     "normalized_confusion_matrix": normalized_confusion_matrix.tolist(),
        #     "cell_img_names": self.cell_img_names,
        # }
        # return final_result

    def evaluate_sample(self, pred, gt, img_name):
        """Runs the analysis for a single sample.
        :param pred: Predicted segmentation as a numpy array of shape (H,W).
        :param gt: Ground-truth segmentation as a numpy array of shape (H,W).
        :returns: Dictionary holding results for this sample.
        """
        if pred.shape != gt.shape:
            raise RuntimeError(
                f"Shapes of prediction and annotation do not match! Pred: {pred.shape}, GT: {gt.shape}"
            )
        H, W = pred.shape
        results = np.full(
            shape=(self.num_classes, H, W),
            fill_value=ERROR_CODES["unassigned"],
            dtype=np.int8,
        )
        # IGNORE
        if self.ignore_index:
            ignore_inds_y, ignore_inds_x = np.where(gt == self.ignore_index)
            results[:, ignore_inds_y, ignore_inds_x] = ERROR_CODES["ignore"]

        pred_one_hot = one_hot(pred, num_classes=self.num_classes, ignore_index=self.ignore_index)
        gt_one_hot = one_hot(gt, num_classes=self.num_classes, ignore_index=self.ignore_index)

        # select only the active classes
        if GPU:
            pred_one_hot = np.asarray(pred_one_hot)
            gt_one_hot = np.asarray(gt_one_hot)

        active_mask = np.logical_or(
            pred_one_hot.any(axis=(1, 2)),
            gt_one_hot.any(axis=(1, 2)),
        )
        pred_one_hot_active = pred_one_hot[active_mask]
        gt_one_hot_active = gt_one_hot[active_mask]
        pred_active = np.argmax(pred_one_hot_active, axis=0)
        gt_active = np.argmax(gt_one_hot_active, axis=0)
        if self.ignore_index:
            gt_active[ignore_inds_y, ignore_inds_x] = self.ignore_index
        results_active = results[active_mask]
        results_inactive = results[~active_mask]

        # TRUE POSITIVE
        tp_mask = np.logical_and(pred_one_hot_active, gt_one_hot_active)
        results_active[tp_mask] = ERROR_CODES["TP"]

        # TRUE NEGATIVE
        # active classes
        tn_mask = ~np.logical_or(pred_one_hot_active, gt_one_hot_active)
        results_on_mask = results_active[tn_mask]
        results_active[tn_mask] = np.where(
            results_on_mask != ERROR_CODES["unassigned"],
            results_on_mask,
            ERROR_CODES["TN"],
        )
        # inactive classes (everything that is not ignore is TN)
        results_inactive[results_inactive == ERROR_CODES["unassigned"]] = ERROR_CODES["TN"]

        # FALSE POSITIVE
        fp_mask = np.logical_and(pred_one_hot_active, ~gt_one_hot_active)

        # FALSE NEGATIVE
        fn_mask = np.logical_and(~pred_one_hot_active, gt_one_hot_active)

        # BOUNDARY
        results_active = self.get_boundary_errors(
            results=results_active,
            tp_mask=tp_mask,
            tn_mask=tn_mask,
            fp_mask=fp_mask,
            fn_mask=fn_mask,
        )

        # EXTENT / SEGMENT
        results_active = self.get_extent_segment_errors(
            results=results_active,
            pred_one_hot=pred_one_hot_active,
            gt_one_hot=gt_one_hot_active,
        )

        results[active_mask] = results_active
        results[~active_mask] = results_inactive
        assert not (results == ERROR_CODES["unassigned"]).any()

        # Boundary IoU
        if self.ignore_index:
            ignore_inds = (ignore_inds_y, ignore_inds_x)
        else:
            ignore_inds = None
        (
            boundary_intersection_counts_active,
            boundary_union_counts_active,
        ) = self.evaluate_sample_boundary_iou(
            sample_results=results_active,
            pred_one_hot=pred_one_hot_active,
            gt_one_hot=gt_one_hot_active,
            ignore_inds=ignore_inds,
        )
        boundary_intersection_counts = np.zeros(self.num_classes, dtype=np.int64)
        boundary_union_counts = np.zeros(self.num_classes, dtype=np.int64)

        boundary_intersection_counts[active_mask] += boundary_intersection_counts_active
        boundary_union_counts[active_mask] += boundary_union_counts_active

        return dict(
            main_results=results,
            boundary_iou_results=(boundary_intersection_counts, boundary_union_counts),
        )

    def update_results(self, sample_results, img_name):
        # main results
        image_stats = {}
        for error_name, error_code in ERROR_CODES.items():
            error_values = (sample_results["main_results"] == error_code).sum(axis=(1, 2))
            self.results[error_name] += error_values
            image_stats[error_name] = error_values

        # boundary IoU
        boundary_intersection_counts, boundary_union_counts = sample_results["boundary_iou_results"]
        self.boundary_iou_intersection_counts += boundary_intersection_counts
        self.boundary_iou_union_counts += boundary_union_counts
        image_stats["boundary_iou_intersection_counts"] = boundary_intersection_counts
        image_stats["boundary_iou_union_counts"] = boundary_union_counts
        self.calculate_per_image_metrics(image_stats, img_name)

    def calculate_per_image_metrics(self, image_stats, img_name):
        fp = image_stats["FP_boundary"] + image_stats["FP_extent"] + image_stats["FP_segment"]
        fn = image_stats["FN_boundary"] + image_stats["FN_extent"] + image_stats["FN_segment"]
        tp = image_stats["TP"]
        tn = image_stats["TN"]

        e_boundary = image_stats["FP_boundary"] + image_stats["FN_boundary"]
        e_extent = image_stats["FP_extent"] + image_stats["FN_extent"]
        e_segment = image_stats["FP_segment"] + image_stats["FN_segment"]

        union = tp + fp + fn
        iou = tp / union

        overall_tp = tp[: self.num_classes].sum()
        overall_fn = fn[: self.num_classes].sum()
        pixel_acc = overall_tp / (overall_tp + overall_fn)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 / (1.0 / precision + 1.0 / recall)

        fp_boundary_ou = image_stats["FP_boundary"] / union
        fn_boundary_ou = image_stats["FN_boundary"] / union
        e_boundary_ou = e_boundary / union

        fp_extent_ou = image_stats["FP_extent"] / union
        fn_extent_ou = image_stats["FN_extent"] / union
        e_extent_ou = e_extent / union

        fp_segment_ou = image_stats["FP_segment"] / union
        fn_segment_ou = image_stats["FN_segment"] / union
        e_segment_ou = e_segment / union

        e_boundary_ou_renormed = e_boundary / (tp + e_boundary)
        e_extent_ou_renormed = e_extent / (tp + e_boundary + e_extent)
        e_segment_ou_renormed = e_segment_ou

        with numpy.errstate(invalid="ignore"):
            boundary_iou = (
                image_stats["boundary_iou_intersection_counts"]
                / image_stats["boundary_iou_union_counts"]
            )

        def postprocess_values(values):
            values = values[~numpy.isnan(values)]
            value = round(float(np.mean(values)), 2)
            return value

        self.image_metrics["pixel_acc"].append(round(float(pixel_acc), 2))
        self.image_metrics["precision"].append(postprocess_values(precision))
        self.image_metrics["recall"].append(postprocess_values(recall))
        self.image_metrics["f1_score"].append(postprocess_values(f1_score))
        self.image_metrics["iou"].append(postprocess_values(iou))
        self.image_metrics["boundary_iou"].append(postprocess_values(boundary_iou))
        self.image_metrics["boundary_eou"].append(postprocess_values(e_boundary_ou))
        self.image_metrics["extent_eou"].append(postprocess_values(e_extent_ou))
        self.image_metrics["segment_eou"].append(postprocess_values(e_segment_ou))
        self.image_metrics["boundary_eou_renormed"].append(
            postprocess_values(e_boundary_ou_renormed)
        )
        self.image_metrics["extent_eou_renormed"].append(postprocess_values(e_extent_ou_renormed))
        self.image_metrics["segment_eou_renormed"].append(postprocess_values(e_segment_ou_renormed))

        self.img_names.append(img_name)

    def get_boundary_errors(self, results, tp_mask, tn_mask, fp_mask, fn_mask):
        H, W = tp_mask.shape[-2:]
        if self.use_relative_boundary_width:
            img_diag = np.sqrt(H**2 + W**2)
            if GPU:
                img_diag = img_diag.get()
                tp_mask = tp_mask.get()
                tn_mask = tn_mask.get()

            boundary_width = int(round(self.boundary_width * img_diag))
        else:
            boundary_width = self.boundary_width

        tp_ext_boundary = get_exterior_boundary(
            tp_mask, width=boundary_width, implementation=self.boundary_implementation
        )
        tn_ext_boundary = get_exterior_boundary(
            tn_mask, width=boundary_width, implementation=self.boundary_implementation
        )

        if GPU:
            tp_ext_boundary, tn_ext_boundary = np.asarray(tp_ext_boundary), np.asarray(
                tn_ext_boundary
            )

        boundary_intersection = np.logical_and(tp_ext_boundary, tn_ext_boundary)
        fp_boundary_mask_naive = np.logical_and(fp_mask, boundary_intersection)
        fn_boundary_mask_naive = np.logical_and(fn_mask, boundary_intersection)

        if GPU:
            fp_boundary_mask_naive, fn_boundary_mask_naive = (
                fp_boundary_mask_naive.get(),
                fn_boundary_mask_naive.get(),
            )

        dilated_fp_boundary_mask = dilate_mask(
            mask=fp_boundary_mask_naive,
            width=boundary_width,
            implementation=self.boundary_implementation,
        )
        dilated_fn_boundary_mask = dilate_mask(
            mask=fn_boundary_mask_naive,
            width=boundary_width,
            implementation=self.boundary_implementation,
        )

        if GPU:
            dilated_fp_boundary_mask = np.asarray(dilated_fp_boundary_mask)
            dilated_fn_boundary_mask = np.asarray(dilated_fn_boundary_mask)

        fp_boundary_mask = np.logical_and(dilated_fp_boundary_mask, fp_mask)
        fn_boundary_mask = np.logical_and(dilated_fn_boundary_mask, fn_mask)

        if GPU:
            fp_boundary_mask = fp_boundary_mask.get()
            fn_boundary_mask = fn_boundary_mask.get()

        # check if every segment of boundary errors has a TP and a TN as direct neighbor
        fp_boundary_segments = get_contiguous_segments(fp_boundary_mask)
        fn_boundary_segments = get_contiguous_segments(fn_boundary_mask)

        tp_contour = get_exterior_boundary(tp_mask, width=1, implementation="fast")
        tn_contour = get_exterior_boundary(tn_mask, width=1, implementation="fast")

        for c, boundary_segments in fp_boundary_segments.items():
            if c == self.ignore_index:
                continue
            for segment in boundary_segments:
                if (not tp_contour[c][segment].any()) or (not tn_contour[c][segment].any()):
                    fp_boundary_mask[c][segment] = False

        for c, boundary_segments in fn_boundary_segments.items():
            if c == self.ignore_index:
                continue
            for segment in boundary_segments:
                if (not tp_contour[c][segment].any()) or (not tn_contour[c][segment].any()):
                    fn_boundary_mask[c][segment] = False

        results_on_mask = results[fp_boundary_mask]
        results[fp_boundary_mask] = np.where(
            results_on_mask != ERROR_CODES["unassigned"],
            results_on_mask,
            ERROR_CODES["FP_boundary"],
        )
        results_on_mask = results[fn_boundary_mask]
        results[fn_boundary_mask] = np.where(
            results_on_mask != ERROR_CODES["unassigned"],
            results_on_mask,
            ERROR_CODES["FN_boundary"],
        )
        return results

    def get_extent_segment_errors(
        self,
        results,
        pred_one_hot,
        gt_one_hot,
    ):
        if GPU:
            pred_one_hot = pred_one_hot.get()
            gt_one_hot = gt_one_hot.get()

        pred_segments = get_contiguous_segments(pred_one_hot)
        gt_segments = get_contiguous_segments(gt_one_hot)

        for c, (pred_c, gt_c) in enumerate(zip(pred_one_hot, gt_one_hot)):
            if pred_c.any():
                if gt_c.any():
                    # positve
                    for pred_segment in pred_segments[c]:
                        results_on_segment = results[c][pred_segment]
                        if (results_on_segment == ERROR_CODES["unassigned"]).any():
                            error_type = (
                                "FP_extent"
                                if (results_on_segment == ERROR_CODES["TP"]).any()
                                else "FP_segment"
                            )
                            results[c][pred_segment] = np.where(
                                results_on_segment != ERROR_CODES["unassigned"],
                                results_on_segment,
                                ERROR_CODES[error_type],
                            )

                    # negative
                    for gt_segment in gt_segments[c]:
                        results_on_segment = results[c][gt_segment]
                        if (results_on_segment == ERROR_CODES["unassigned"]).any():
                            error_type = (
                                "FN_extent"
                                if (results_on_segment == ERROR_CODES["TP"]).any()
                                else "FN_segment"
                            )
                            results[c][gt_segment] = np.where(
                                results_on_segment != ERROR_CODES["unassigned"],
                                results_on_segment,
                                ERROR_CODES[error_type],
                            )
                else:  # only FP segment errors for this class
                    # positive prediction must be a superset of unassigned
                    # every prediction can only be unassigned or ignore
                    if GPU:
                        pred_c = np.asarray(pred_c)
                    assert pred_c[results[c] == ERROR_CODES["unassigned"]].all()
                    results[c][results[c] == ERROR_CODES["unassigned"]] = ERROR_CODES["FP_segment"]
            else:
                if gt_c.any():  # only FN segment errors for this class
                    results[c][results[c] == ERROR_CODES["unassigned"]] = ERROR_CODES["FN_segment"]
                else:
                    continue

        return results

    def evaluate_sample_boundary_iou(
        self, sample_results, pred_one_hot, gt_one_hot, ignore_inds=None
    ):
        H, W = sample_results.shape[-2:]
        img_diag = np.sqrt(H**2 + W**2)

        if GPU:
            img_diag = img_diag.get()
            pred_one_hot = pred_one_hot.get()
            gt_one_hot = gt_one_hot.get()

        boundary_width = max(int(round(self.boundary_iou_d * img_diag)), 1)

        # BoundaryIoU uses "fast" boundary implementation, see https://github.com/bowenc0221/boundary-iou-api/blob/37d25586a677b043ed585f10e5c42d4e80176ea9/boundary_iou/utils/boundary_utils.py#L12
        pred_one_hot_int_boundary = get_interior_boundary(
            pred_one_hot, width=boundary_width, implementation="fast"
        )  # P_d ∩ P
        gt_one_hot_int_boundary = get_interior_boundary(
            gt_one_hot, width=boundary_width, implementation="fast"
        )  # G_d ∩ G
        gt_one_hot_ext_boundary = get_exterior_boundary(
            gt_one_hot, width=boundary_width, implementation="fast"
        )  # G_d - G

        if GPU:
            pred_one_hot_int_boundary = np.asarray(pred_one_hot_int_boundary)
            gt_one_hot_int_boundary = np.asarray(gt_one_hot_int_boundary)

        boundary_intersection = np.logical_and(pred_one_hot_int_boundary, gt_one_hot_int_boundary)
        boundary_union = np.logical_or(pred_one_hot_int_boundary, gt_one_hot_int_boundary)

        if ignore_inds:  # remove ignore pixels
            ignore_inds_y, ignore_inds_x = ignore_inds
            assert not gt_one_hot[:, ignore_inds_y, ignore_inds_x].any()
            boundary_intersection[:, ignore_inds_y, ignore_inds_x] = 0
            boundary_union[:, ignore_inds_y, ignore_inds_x] = 0

        boundary_intersection_counts = boundary_intersection.sum(axis=(1, 2))
        boundary_union_counts = boundary_union.sum(axis=(1, 2))

        return (
            boundary_intersection_counts,
            boundary_union_counts,
        )

    def visualize_single_sample(self, pred, gt, output_dir):
        sample_results = self.evaluate_sample(pred, gt)["main_results"]
        os.makedirs(output_dir, exist_ok=True)
        active_classes = np.unique(np.concatenate([pred, gt]))
        H, W = sample_results.shape[-2:]

        for c in active_classes:
            if c == self.ignore_index:
                continue
            pred_c = (pred == c).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(output_dir, f"{self.class_names[c]}_pred.png"), pred_c)
            gt_c = (gt == c).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(output_dir, f"{self.class_names[c]}_gt.png"), gt_c)
            error_map = np.zeros((H, W, 3), dtype=np.uint8)
            for error_type, error_color in ERROR_PALETTE.items():
                error_map[sample_results[c] == error_type] = error_color
            error_map = cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(output_dir, f"{self.class_names[c]}_errors.png"), error_map)
        print(f"Saved visualization to {output_dir}.")
        return

    def calculate_error_metrics(self):
        dataframe = pd.DataFrame(index=self.class_names)
        for error_name, error_counts in self.results.items():
            if error_name == "unassigned":
                assert (error_counts == 0).all()
                continue
            dataframe[error_name] = error_counts

        dataframe["FP"] = (
            dataframe["FP_boundary"] + dataframe["FP_extent"] + dataframe["FP_segment"]
        )
        dataframe["FN"] = (
            dataframe["FN_boundary"] + dataframe["FN_extent"] + dataframe["FN_segment"]
        )
        dataframe["E_boundary"] = dataframe["FP_boundary"] + dataframe["FN_boundary"]
        dataframe["E_extent"] = dataframe["FP_extent"] + dataframe["FN_extent"]
        dataframe["E_segment"] = dataframe["FP_segment"] + dataframe["FN_segment"]

        union = dataframe["TP"] + dataframe["FP"] + dataframe["FN"]
        dataframe["IoU"] = dataframe["TP"] / union
        dataframe["precision"] = dataframe["TP"] / (dataframe["TP"] + dataframe["FP"])
        dataframe["recall"] = dataframe["TP"] / (dataframe["TP"] + dataframe["FN"])
        dataframe["F1_score"] = 2 / (1.0 / dataframe["precision"] + 1.0 / dataframe["recall"])

        dataframe["FP_boundary_oU"] = dataframe["FP_boundary"] / union
        dataframe["FN_boundary_oU"] = dataframe["FN_boundary"] / union
        dataframe["E_boundary_oU"] = dataframe["E_boundary"] / union

        dataframe["FP_extent_oU"] = dataframe["FP_extent"] / union
        dataframe["FN_extent_oU"] = dataframe["FN_extent"] / union
        dataframe["E_extent_oU"] = dataframe["E_extent"] / union

        dataframe["FP_segment_oU"] = dataframe["FP_segment"] / union
        dataframe["FN_segment_oU"] = dataframe["FN_segment"] / union
        dataframe["E_segment_oU"] = dataframe["E_segment"] / union

        dataframe["E_boundary_oU_renormed"] = dataframe["E_boundary"] / (
            dataframe["TP"] + dataframe["E_boundary"]
        )
        dataframe["E_extent_oU_renormed"] = dataframe["E_extent"] / (
            dataframe["TP"] + dataframe["E_boundary"] + dataframe["E_extent"]
        )
        dataframe["E_segment_oU_renormed"] = dataframe["E_segment_oU"]

        with np.errstate(invalid="ignore"):  # avoid warnings for zero-division
            # boundary IoU
            dataframe["boundary_IoU"] = (
                self.boundary_iou_intersection_counts / self.boundary_iou_union_counts
            )
            # aggregate classes
            dataframe.loc["mean"] = dataframe.mean(axis=0)

        # dataframe.to_csv(f"{evaluator.result_dir}/result_df.csv", index=True)

        return dataframe
