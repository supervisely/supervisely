from typing import List

from supervisely.nn.benchmark.visualization.evaluation_result import EvalResult
from supervisely.nn.benchmark.visualization.widgets import GalleryWidget


class BaseVisMetric:

    def __init__(
        self,
        vis_texts,
        eval_results: List[EvalResult],
        explore_modal_table: GalleryWidget = None,
        diff_modal_table: GalleryWidget = None,
    ) -> None:
        self.vis_texts = vis_texts
        self.eval_results = eval_results
        self.explore_modal_table = explore_modal_table
        self.diff_modal_table = diff_modal_table
