from typing import Union, List

import supervisely as sly
from supervisely.nn.inference import SessionJSON
from supervisely.nn.benchmark.evaluation import BaseEvaluator
from supervisely.nn.benchmark.evaluation.object_detection import calculate_metrics, MetricProvider


class ObjectDetectionEvaluator(BaseEvaluator):
    def evaluate(self):
        eval_data = calculate_metrics(cocoGt, cocoDt)

    def run_inference(self, batch_size: int = 8, cache_project: bool = False):
        try_set_conf_auto(self.session, CONF_THRES)
        super().run_inference(batch_size, cache_project)

    
def try_set_conf_auto(session: SessionJSON, conf: float):
    conf_names = ["conf", "confidence", "confidence_threshold"]
    default = session.get_default_inference_settings()
    for name in conf_names:
        if name in default:
            session.inference_settings[name] = conf
            return True
    return False
    