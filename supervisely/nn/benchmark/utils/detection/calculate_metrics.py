from collections import defaultdict
from typing import Callable, List, Literal, Optional

import numpy as np

from supervisely.nn.benchmark.utils.detection.coco_eval import (
    COCO,
    SlyCOCOeval,
    pycocotools_installed,
)


def set_cocoeval_params(
    cocoeval: SlyCOCOeval,
    parameters: dict,
):
    """
    type cocoeval: COCOeval
    """
    if parameters is None:
        return
    param_names = (
        "iouThrs",
        "recThrs",
        "maxDets",
        "areaRng",
        "areaRngLbl",
        # "kpt_oks_sigmas" # For keypoints
    )
    for param_name in param_names:
        cocoeval.params.__setattr__(
            param_name, parameters.get(param_name, cocoeval.params.__getattribute__(param_name))
        )


def calculate_metrics(
    cocoGt: COCO,
    cocoDt: COCO,
    iouType: Literal["bbox", "segm"],
    progress_cb: Optional[Callable] = None,
    evaluation_params: Optional[dict] = None,
):
    """
    Calculate COCO metrics.

    :param cocoGt: Ground truth dataset in COCO format
    :type cocoGt: COCO
    :param cocoDt: Predicted dataset in COCO format
    :type cocoDt: COCO
    :param iouType: Type of IoU calculation
    :type iouType: Literal["bbox", "segm"]
    :param progress_cb: Progress callback
    :type progress_cb: Optional[Callable]
    :return: Results of the evaluation
    :rtype: dict
    """
    if not pycocotools_installed:
        raise ImportError("pycocotools is not installed")

    evaluation_params = evaluation_params or {}
    max_dets = evaluation_params.get("max_detections", 100)

    cocoEval = SlyCOCOeval(cocoGt, cocoDt, iouType=iouType)
    cocoEval.params.maxDets[-1] = max_dets
    cocoEval.evaluate()
    progress_cb(1) if progress_cb is not None else None
    cocoEval.accumulate()
    progress_cb(1) if progress_cb is not None else None
    cocoEval.summarize()

    # For classification metrics
    cocoEval_cls = SlyCOCOeval(cocoGt, cocoDt, iouType=iouType)
    cocoEval_cls.params.useCats = 0
    cocoEval_cls.params.maxDets[-1] = max_dets
    cocoEval_cls.evaluate()
    progress_cb(1) if progress_cb is not None else None
    cocoEval_cls.accumulate()
    progress_cb(1) if progress_cb is not None else None
    cocoEval_cls.summarize()

    iouThrs = cocoEval.params.iouThrs
    iou_threshold = evaluation_params.get("iou_threshold", 0.5)
    iou_threshold_per_class = evaluation_params.get("iou_threshold_per_class")
    if iou_threshold_per_class is not None:
        iou_idx_per_class = {
            cocoGt.getCatIds(catNms=[class_name])[0]: np.where(np.isclose(iouThrs, iou_thres))[0][0]
            for class_name, iou_thres in iou_threshold_per_class.items()
        }
    else:
        iou_idx = np.where(np.isclose(iouThrs, iou_threshold))[0][0]
        iou_idx_per_class = {cat_id: iou_idx for cat_id in cocoGt.getCatIds()}

    # TODO: Add support for average_across_iou_thresholds
    if iou_threshold_per_class is not None or iou_threshold != 0.5:
        average_across_iou_thresholds = False
        evaluation_params["average_across_iou_thresholds"] = average_across_iou_thresholds

    eval_img_dict = get_eval_img_dict(cocoEval)
    eval_img_dict_cls = get_eval_img_dict(cocoEval_cls)
    matches = get_matches(
        eval_img_dict,
        eval_img_dict_cls,
        cocoEval_cls,
        iou_idx_per_class=iou_idx_per_class,
    )
    # true_positives, false_positives, false_negatives = get_counts(eval_img_dict, cocoEval_cls)

    params = {
        "iouThrs": cocoEval.params.iouThrs,
        "recThrs": cocoEval.params.recThrs,
        "evaluation_params": evaluation_params,
        "iou_idx_per_class": iou_idx_per_class,
    }
    coco_metrics = {"mAP": cocoEval.stats[0], "precision": cocoEval.eval["precision"]}
    coco_metrics["AP50"] = cocoEval.stats[1]
    coco_metrics["AP75"] = cocoEval.stats[2]
    eval_data = {
        "matches": matches,
        "coco_metrics": coco_metrics,
        "params": params,
    }
    progress_cb(1) if progress_cb is not None else None

    return eval_data


def get_counts(eval_img_dict: dict, cocoEval_cls: SlyCOCOeval):
    if not pycocotools_installed:
        raise ImportError("pycocotools is not installed")

    cat_ids = cocoEval_cls.cocoGt.getCatIds()
    iouThrs = cocoEval_cls.params.iouThrs
    catId2idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    true_positives = np.zeros((len(cat_ids), len(iouThrs)))
    false_positives = np.zeros((len(cat_ids), len(iouThrs)))
    false_negatives = np.zeros((len(cat_ids), len(iouThrs)))
    for img_id, eval_imgs in eval_img_dict.items():
        for eval_img in eval_imgs:
            cat_id = eval_img["category_id"]
            cat_idx = catId2idx[cat_id]
            gtIgnore = eval_img["gtIgnore"]
            # if conf_thresh is not None:
            #     scores = np.array(eval_img["dtScores"])
            #     dt_conf_mask = scores < conf_thresh
            #     dt_not_ignore[:, dt_conf_mask] = False
            #     for idx, dt_id in enumerate(eval_img['dtIds']):
            #         if dt_conf_mask[idx]:
            #             eval_img["gtMatches"][eval_img["gtMatches"] == dt_id] = 0.
            gt_not_ignore_idxs = np.where(np.logical_not(gtIgnore))[0]
            dt_not_ignore = np.logical_not(eval_img["dtIgnore"])
            true_positives[cat_idx] += ((eval_img["dtMatches"] > 0) & dt_not_ignore).sum(1)
            false_positives[cat_idx] += ((eval_img["dtMatches"] == 0) & dt_not_ignore).sum(1)
            false_negatives[cat_idx] += (eval_img["gtMatches"][:, gt_not_ignore_idxs] == 0).sum(1)
    return true_positives.astype(int), false_positives.astype(int), false_negatives.astype(int)


def get_counts_and_scores(cocoEval: SlyCOCOeval, cat_id: int, t: int):
    """Returns tps, fps, scores, n_positives"""

    if not pycocotools_installed:
        raise ImportError("pycocotools is not installed")

    aRng = cocoEval.params.areaRng[0]
    eval_imgs = [ev for ev in cocoEval.evalImgs if ev is not None and ev["aRng"] == aRng]

    tps = []
    fps = []
    # fns = []
    scores = []
    n_positives = 0

    # Process each evaluated image
    for eval_img in eval_imgs:
        if eval_img["category_id"] != cat_id:
            continue
        dtScores = eval_img["dtScores"]
        dtm = eval_img["dtMatches"][t]
        gtm = eval_img["gtMatches"][t]

        # ntp = (dtm > 0).sum()
        # nfp = (dtm == 0).sum()
        # nfn = (gtm == 0).sum()
        p = len(gtm)

        tp = (dtm > 0).astype(int).tolist()
        fp = (dtm == 0).astype(int).tolist()
        # fn = [nfn]*len(dtm)

        tps.extend(tp)
        fps.extend(fp)
        # fns.extend(fn)
        scores.extend(dtScores)
        n_positives += p

    assert len(tps) == len(fps) == len(scores)

    # sort by score
    indices = np.argsort(scores)[::-1]
    scores = np.array(scores)[indices]
    tps = np.array(tps)[indices]
    fps = np.array(fps)[indices]

    return tps, fps, scores, n_positives


def get_eval_img_dict(cocoEval: SlyCOCOeval):
    if not pycocotools_installed:
        raise ImportError("pycocotools is not installed")

    aRng = cocoEval.params.areaRng[0]
    eval_img_dict = defaultdict(list)  # img_id : dt/gt
    for i, eval_img in enumerate(cocoEval.evalImgs):
        if eval_img is None or eval_img["aRng"] != aRng:
            continue
        img_id = eval_img["image_id"]
        cat_id = eval_img["category_id"]
        ious = cocoEval.ious[(img_id, cat_id)]
        # ! inplace operation
        eval_img["ious"] = ious
        eval_img_dict[img_id].append(eval_img)
    eval_img_dict = dict(eval_img_dict)
    return eval_img_dict


def _get_missclassified_match(eval_img_cls: SlyCOCOeval, dt_id, gtIds_orig, dtIds_orig, iou_t):
    if not pycocotools_installed:
        raise ImportError("pycocotools is not installed")

    # Correction on miss-classification
    gt_idx = np.nonzero(eval_img_cls["gtMatches"][iou_t] == dt_id)[0]
    if len(gt_idx) == 1:
        gt_idx = gt_idx[0]
        gt_id = eval_img_cls["gtIds"][gt_idx]
        gt_idx_o = gtIds_orig.index(gt_id)
        dt_idx_o = dtIds_orig.index(dt_id)
        iou = eval_img_cls["ious"][dt_idx_o, gt_idx_o].item()
        is_gt_ignore = eval_img_cls["gtIgnore"][gt_idx]
        if not is_gt_ignore:
            return gt_id, iou
    elif len(gt_idx) > 1:
        raise ValueError("Multiple matches")
    return None, None


def get_matches(
    eval_img_dict: dict,
    eval_img_dict_cls: dict,
    cocoEval_cls: SlyCOCOeval,
    iou_idx_per_class: dict = None,
):
    if not pycocotools_installed:
        raise ImportError("pycocotools is not installed")

    cat_ids = cocoEval_cls.cocoGt.getCatIds()
    matches = []
    for img_id, eval_imgs in eval_img_dict.items():

        # get miss-classified
        eval_img_cls = eval_img_dict_cls[img_id][0]
        gt_ids_orig_cls = [_["id"] for i in cat_ids for _ in cocoEval_cls._gts[img_id, i]]

        for eval_img in eval_imgs:
            cat_id = eval_img["category_id"]
            iou_t = iou_idx_per_class[cat_id]
            dtIds = np.array(eval_img["dtIds"])
            gtIds = np.array(eval_img["gtIds"])
            dtm = eval_img["dtMatches"][iou_t]
            gtm = eval_img["gtMatches"][iou_t]
            dtIgnore = eval_img["dtIgnore"][iou_t]
            gtIgnore = eval_img["gtIgnore"]

            # True Positives
            tp_idxs = np.nonzero(dtm)[0]
            for i in tp_idxs:
                if dtIgnore[i]:
                    continue
                dt_id = dtIds[i]
                gt_id = int(dtm[i])
                gt_idx = np.where(gtIds == gt_id)[0]
                iou = eval_img["ious"][i, gt_idx].item()
                score = eval_img["dtScores"][i]
                match = {
                    "image_id": eval_img["image_id"],
                    "category_id": eval_img["category_id"],
                    "dt_id": dt_id,
                    "gt_id": gt_id,
                    "type": "TP",
                    "score": score,
                    "iou": iou,
                    "miss_cls": False,
                }
                assert iou >= 0.5, iou
                matches.append(match)

            # False Positives
            fp_idxs = np.nonzero(dtm == 0)[0]
            for i in fp_idxs:
                dt_id = dtIds[i]
                score = eval_img["dtScores"][i]
                match = {
                    "image_id": eval_img["image_id"],
                    "category_id": eval_img["category_id"],
                    "dt_id": dt_id,
                    "gt_id": None,
                    "type": "FP",
                    "score": score,
                    "iou": None,
                    "miss_cls": False,
                }

                # Correction on miss-classification
                cls_gt_id, iou = _get_missclassified_match(
                    eval_img_cls, dt_id, gt_ids_orig_cls, eval_img_cls["dtIds"], iou_t
                )
                if cls_gt_id is not None:
                    assert iou >= 0.5, iou
                    match["gt_id"] = cls_gt_id
                    match["iou"] = iou
                    match["miss_cls"] = True
                matches.append(match)

            # False Negatives
            fn_idxs = np.nonzero(gtm == 0)[0]
            for i in fn_idxs:
                if gtIgnore[i]:
                    continue
                gt_id = gtIds[i]
                match = {
                    "image_id": eval_img["image_id"],
                    "category_id": eval_img["category_id"],
                    "dt_id": None,
                    "gt_id": gt_id,
                    "type": "FN",
                    "score": None,
                    "iou": None,
                    "miss_cls": False,
                }
                matches.append(match)

    return matches


def get_rare_classes(cocoGt: COCO, topk_ann_fraction=0.1, topk_classes_fraction=0.2):
    if not pycocotools_installed:
        raise ImportError("pycocotools is not installed")

    anns_cat_ids = [ann["category_id"] for ann in cocoGt.anns.values()]
    cat_ids, cat_counts = np.unique(anns_cat_ids, return_counts=True)
    inds_sorted = np.argsort(cat_counts)
    cum_counts = cat_counts[inds_sorted].cumsum()

    topk_upper1 = np.sum(cum_counts < cum_counts[-1] * topk_ann_fraction)
    topk_upper2 = int(topk_classes_fraction * len(cat_counts))
    topk_lower1 = 1

    topk = max(topk_lower1, min(topk_upper1, topk_upper2))
    cat_ids_rare = cat_ids[inds_sorted][:topk]
    cat_names = [cocoGt.cats[cat_id]["name"] for cat_id in cat_ids_rare]
    return cat_ids_rare, cat_names
