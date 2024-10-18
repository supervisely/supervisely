from supervisely.io.json import load_json_file


def read_coco_datasets(cocoGt, cocoDt):
    from pycocotools.coco import COCO  # pylint: disable=import-error

    if isinstance(cocoGt, str) and isinstance(cocoDt, str):
        cocoGt = COCO(cocoGt)
        cocoDt = COCO(cocoDt)
        return cocoGt, cocoDt
    cocoGt = COCO()
    cocoGt.dataset = cocoGt
    cocoGt.createIndex()
    cocoDt = cocoGt.loadRes(cocoDt["annotations"])
    return cocoGt, cocoDt
