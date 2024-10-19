from supervisely.io.json import load_json_file


def read_coco_datasets(cocoGt, cocoDt):
    from pycocotools.coco import COCO  # pylint: disable=import-error

    if isinstance(cocoGt, str) and isinstance(cocoDt, str):
        cocoGt = COCO(cocoGt)
        cocoDt = COCO(cocoDt)
        return 
    gt = COCO()
    gt.dataset = cocoGt
    gt.createIndex()
    dt = gt.loadRes(cocoDt["annotations"])
    return gt, dt
