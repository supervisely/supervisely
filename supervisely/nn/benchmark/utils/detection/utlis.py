from supervisely.io.json import load_json_file
from supervisely.nn.inference import SessionJSON


def try_set_conf_auto(session: SessionJSON, conf: float):
    conf_names = ["conf", "confidence", "confidence_threshold", "confidence_thresh"]
    default = session.get_default_inference_settings()
    for name in conf_names:
        if name in default:
            session.inference_settings[name] = conf
            return True
    return False


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
