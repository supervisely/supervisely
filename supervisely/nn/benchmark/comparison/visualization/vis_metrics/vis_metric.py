from typing import List

from supervisely.nn.benchmark.comparison.evaluation_result import EvalResult
from supervisely.nn.benchmark.comparison.visualization.widgets.widget import BaseWidget


class BaseVisMetric:
    def __init__(self, vis_texts, eval_results: List[EvalResult]) -> None:
        self.vis_texts = vis_texts
        self.eval_results = eval_results
