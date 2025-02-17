import numpy as np

try:
    from pycocotools.coco import COCO  # pylint: disable=import-error
    from pycocotools.cocoeval import COCOeval  # pylint: disable=import-error
except ImportError:
    COCOeval = None
    COCO = None


class SlyCOCOeval:
    def summarize(coco_eval_obj):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = coco_eval_obj.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = coco_eval_obj.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = coco_eval_obj.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1, maxDets=coco_eval_obj.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=0.5, maxDets=coco_eval_obj.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=coco_eval_obj.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", maxDets=coco_eval_obj.params.maxDets[2])
            stats[4] = _summarize(1, areaRng="medium", maxDets=coco_eval_obj.params.maxDets[2])
            stats[5] = _summarize(1, areaRng="large", maxDets=coco_eval_obj.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=coco_eval_obj.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=coco_eval_obj.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=coco_eval_obj.params.maxDets[2])
            stats[9] = _summarize(0, areaRng="small", maxDets=coco_eval_obj.params.maxDets[2])
            stats[10] = _summarize(0, areaRng="medium", maxDets=coco_eval_obj.params.maxDets[2])
            stats[11] = _summarize(0, areaRng="large", maxDets=coco_eval_obj.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=20, areaRng="medium")
            stats[4] = _summarize(1, maxDets=20, areaRng="large")
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=20, areaRng="medium")
            stats[9] = _summarize(0, maxDets=20, areaRng="large")
            return stats

        if not coco_eval_obj.eval:
            raise Exception("Please run accumulate() first")
        iouType = coco_eval_obj.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        coco_eval_obj.stats = summarize()
