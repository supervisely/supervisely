from supervisely.io.json import load_json_file


def read_coco_datasets(cocoGt_json, cocoDt_json):
    from pycocotools.coco import COCO  # pylint: disable=import-error

    if isinstance(cocoGt_json, str):
        cocoGt_json = load_json_file(cocoGt_json)
    if isinstance(cocoDt_json, str):
        cocoDt_json = load_json_file(cocoDt_json)
    cocoGt = COCO()
    cocoGt.dataset = cocoGt_json
    cocoGt.createIndex()
    cocoDt = cocoGt.loadRes(cocoDt_json["annotations"])
    return cocoGt, cocoDt
