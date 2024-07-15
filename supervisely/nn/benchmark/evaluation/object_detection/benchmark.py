import os
from typing import Union, List
from tqdm import tqdm

import supervisely as sly
from supervisely.nn.inference import SessionJSON
from supervisely.nn.benchmark.evaluation import BaseBenchmark, BaseEvaluator, ObjectDetectionEvaluator


CONF_THRES = 0.05


class ObjectDetectionBenchmark(BaseBenchmark):
    def _init_evaluator(self) -> BaseEvaluator:
        return ObjectDetectionEvaluator()
    
    def _run_inference(
            self,
            output_project_id=None,
            batch_size: int = 8,
            cache_project: bool = False
            ):
        assert try_set_conf_auto(self.session, CONF_THRES), f"Unable to set the confidence threshold to default value of {CONF_THRES}."
        return super()._run_inference(output_project_id, batch_size, cache_project)
    
        
def try_set_conf_auto(session: SessionJSON, conf: float):
    conf_names = ["conf", "confidence", "confidence_threshold"]
    default = session.get_default_inference_settings()
    for name in conf_names:
        if name in default:
            session.inference_settings[name] = conf
            return True
    return False
    