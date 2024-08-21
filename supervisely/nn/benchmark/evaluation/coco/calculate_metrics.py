from collections import defaultdict
from typing import Callable, Optional, Literal

import numpy as np


def calculate_metrics(cocoGt, cocoDt, iouType: Literal["bbox", "segm"], progress_cb: Optional[Callable] = None):
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
    from pycocotools.cocoeval import COCOeval  # pylint: disable=import-error

    progress_cb(1) if progress_cb is not None else None
    cocoEval = COCOeval(cocoGt, cocoDt, iouType=iouType)
    progress_cb(1) if progress_cb is not None else None
    cocoEval.evaluate()
    progress_cb(1) if progress_cb is not None else None
    cocoEval.accumulate()
    progress_cb(1) if progress_cb is not None else None
    cocoEval.summarize()
    progress_cb(1) if progress_cb is not None else None

    # For classification metrics
    cocoEval_cls = COCOeval(cocoGt, cocoDt, iouType=iouType)
    progress_cb(1) if progress_cb is not None else None
    cocoEval_cls.params.useCats = 0
    cocoEval_cls.evaluate()
    progress_cb(1) if progress_cb is not None else None
    cocoEval_cls.accumulate()
    progress_cb(1) if progress_cb is not None else None
    cocoEval_cls.summarize()
    progress_cb(1) if progress_cb is not None else None

    eval_img_dict = get_eval_img_dict(cocoEval)
    eval_img_dict_cls = get_eval_img_dict(cocoEval_cls)
    matches = get_matches(eval_img_dict, eval_img_dict_cls, cocoEval_cls, iou_t=0)

    params = {"iouThrs": cocoEval.params.iouThrs, "recThrs": cocoEval.params.recThrs}
    coco_metrics = {"mAP": cocoEval.stats[0], "precision": cocoEval.eval["precision"]}
    eval_data = {
        "matches": matches,
        "coco_metrics": coco_metrics,
        "params": params,
    }
    progress_cb(1) if progress_cb is not None else None

    return eval_data


def get_counts(cocoEval):
    """
    true_positives, false_positives, false_negatives

    type cocoEval: COCOeval
    """
    aRng = cocoEval.params.areaRng[0]
    cat_ids = cocoEval.params.catIds
    eval_imgs = [ev for ev in cocoEval.evalImgs if ev is not None and ev["aRng"] == aRng]

    N = len(eval_imgs)
    T = len(cocoEval.params.iouThrs)
    K = max(cat_ids) + 1

    true_positives = np.zeros((K, N, T))
    false_positives = np.zeros((K, N, T))
    false_negatives = np.zeros((K, N, T))

    for i, eval_img in enumerate(eval_imgs):
        catId = eval_img["category_id"]
        dt_matches = eval_img["dtMatches"]
        gt_matches = eval_img["gtMatches"]

        # Ignore
        if np.any(eval_img["gtIgnore"]):
            dt_matches = eval_img["dtMatches"].copy()
            dt_matches[eval_img["dtIgnore"]] = -1

            gt_matches = eval_img["gtMatches"].copy()
            gt_ignore_mask = eval_img["gtIgnore"][None,].repeat(T, axis=0).astype(bool)
            gt_matches[gt_ignore_mask] = -1

        true_positives[catId, i] = np.sum(dt_matches > 0, axis=1)
        false_positives[catId, i] = np.sum(dt_matches == 0, axis=1)
        false_negatives[catId, i] = np.sum(gt_matches == 0, axis=1)

    return true_positives[cat_ids], false_positives[cat_ids], false_negatives[cat_ids]


def get_counts_and_scores(cocoEval, cat_id: int, t: int):
    """
    tps, fps, scores, n_positives

    type cocoEval: COCOeval
    """
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


def get_eval_img_dict(cocoEval):
    """
    type cocoEval: COCOeval
    """
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


def get_eval_img_dict_cls(cocoEval_cls):
    """
    type cocoEval_cls: COCOeval
    """
    # For miss-classification
    aRng = cocoEval_cls.params.areaRng[0]
    eval_img_dict_cls = defaultdict(list)  # img_id : dt/gt
    for i, eval_img in enumerate(cocoEval_cls.evalImgs):
        if eval_img is None or eval_img["aRng"] != aRng:
            continue
        img_id = eval_img["image_id"]
        cat_id = eval_img["category_id"]
        ious = cocoEval_cls.ious[(img_id, cat_id)]
        # ! inplace operation
        eval_img["ious"] = ious
        eval_img_dict_cls[img_id].append(eval_img)
    eval_img_dict_cls = dict(eval_img_dict_cls)
    assert np.all([len(x) == 1 for x in eval_img_dict_cls.values()])
    return eval_img_dict_cls


def _get_missclassified_match(eval_img_cls, dt_id, gtIds_orig, dtIds_orig, iou_t):
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


def get_matches(eval_img_dict: dict, eval_img_dict_cls: dict, cocoEval_cls, iou_t: int = 0):
    """
    type cocoEval_cls: COCOeval
    """
    catIds = cocoEval_cls.cocoGt.getCatIds()
    matches = []
    for img_id, eval_imgs in eval_img_dict.items():

        # get miss-classified
        eval_img_cls = eval_img_dict_cls[img_id][0]
        gtIds_orig_cls = [_["id"] for cId in catIds for _ in cocoEval_cls._gts[img_id, cId]]

        for eval_img in eval_imgs:

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
                    eval_img_cls, dt_id, gtIds_orig_cls, eval_img_cls["dtIds"], iou_t
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


def get_rare_classes(cocoGt, topk_ann_fraction=0.1, topk_classes_fraction=0.2):
    """
    :param cocoGt: Ground truth dataset in COCO format
    :type cocoGt: COCO
    """
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
