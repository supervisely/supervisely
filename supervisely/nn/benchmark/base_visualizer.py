from typing import List

from supervisely.api.api import Api
from supervisely.nn.benchmark.base_evaluator import BaseEvalResult
from supervisely.nn.benchmark.visualization.renderer import Renderer


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
        self.gt_project_info = None

        for eval_result in self.eval_results:
            self._get_eval_project_infos(eval_result)

    def _get_eval_project_infos(self, eval_result):
        if self.gt_project_info is None:
            self.gt_project_info = self.api.project.get_info_by_id(eval_result.gt_project_id)
        eval_result.gt_project_info = self.gt_project_info

        filters = None
        if eval_result.gt_dataset_ids is not None:
            filters = [{"field": "id", "operator": "in", "value": eval_result.gt_dataset_ids}]
        eval_result.gt_dataset_infos = self.api.dataset.get_list(
            eval_result.gt_project_id,
            filters=filters,
            recursive=True,
        )

        train_info = eval_result.train_info
        if train_info:
            train_task_id = train_info.get("app_session_id")
            if train_task_id:
                eval_result.task_info = self.api.task.get_info_by_id(int(train_task_id))

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
