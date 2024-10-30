from typing import List
from supervisely.nn.benchmark.visualization.renderer import Renderer
from supervisely.api.api import Api
from supervisely.nn.benchmark.base_evaluator import BaseEvalResult

class BaseVisualizer:

    def __init__(
        self,
        api: Api,
        eval_results: List[BaseEvalResult],
        workdir="./visualizations",
    ):
        self.api = api
        self.workdir = workdir
        self.eval_result = eval_results[0]  # for evaluation
        self.eval_results = eval_results  # for comparison

        self.renderer = None

    def visualize(self):
        if self.renderer is None:
            layout = self._create_layout()
            self.renderer = Renderer(layout, self.workdir)
        return self.renderer.visualize()

    def upload_results(self, team_id: int, remote_dir: str, progress=None):
        if self.renderer is None:
            raise RuntimeError("Visualize first")
        return self.renderer.upload_results(self.api, team_id, remote_dir, progress)

    def _create_layout(self):
        raise NotImplementedError("Implement this method in a subclass")
