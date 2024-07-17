import os
from typing import Union, List
from tqdm import tqdm

import supervisely as sly
from supervisely.nn.inference import SessionJSON
from supervisely.nn.benchmark.evaluation import BaseBenchmark, ObjectDetectionEvaluator


CONF_THRES = 0.05


class ObjectDetectionBenchmark(BaseBenchmark):
    def _get_evaluator_class(self) -> type:
        return ObjectDetectionEvaluator
    
    def _run_inference(
            self,
            output_project_id=None,
            batch_size: int = 8,
            cache_project_on_agent: bool = False,
            ):
        assert try_set_conf_auto(self.session, CONF_THRES), f"Unable to set the confidence threshold to {CONF_THRES} for evalation."
        return super()._run_inference(output_project_id, batch_size, cache_project_on_agent)
    
        
def try_set_conf_auto(session: SessionJSON, conf: float):
    conf_names = ["conf", "confidence", "confidence_threshold"]
    default = session.get_default_inference_settings()
    for name in conf_names:
        if name in default:
            session.inference_settings[name] = conf
            return True
    return False
    