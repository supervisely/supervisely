def read_coco_datasets(gt, dt):
    from pycocotools.coco import COCO  # pylint: disable=import-error

    if isinstance(gt, str) and isinstance(dt, str):
        return COCO(gt), COCO(dt)
    cocoGt = COCO()
    cocoGt.dataset = gt
    cocoGt.createIndex()
    cocoDt = cocoGt.loadRes(dt["annotations"])
    return cocoGt, cocoDt
