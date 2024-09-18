import supervisely as sly
from supervisely.nn.benchmark.base_benchmark import BaseBenchmark
from supervisely.nn.benchmark.evaluation.semantic_segmentation_evaluator import (
    SemanticSegmentationEvaluator,
)
from supervisely.nn.inference import SessionJSON
from typing import Union, List


class SemanticSegmentationBenchmark(BaseBenchmark):
    def __init__(
        self,
        api: sly.Api,
        gt_project_id: int,
        gt_dataset_ids: List[int] = None,
        output_dir: str = "./benchmark",
    ):
        self.api = api
        self.session: SessionJSON = None
        self.gt_project_id = gt_project_id
        self.gt_dataset_ids = gt_dataset_ids
        self.gt_project_info = api.project.get_info_by_id(gt_project_id)
        self.dt_project_info = None
        self.output_dir = output_dir
        self.team_id = sly.env.team_id()
        self.evaluator = None
        self._eval_inference_info = None
        self._speedtest = None

    def _get_evaluator_class(self) -> type:
        return SemanticSegmentationEvaluator

    def run_evaluation(
        self,
        model_session: Union[int, str, SessionJSON],
        inference_settings=None,
        output_project_id=None,
        batch_size: int = 8,
        cache_project_on_agent: bool = False,
    ):
        self.run_inference(
            model_session=model_session,
            inference_settings=inference_settings,
            output_project_id=output_project_id,
            batch_size=batch_size,
            cache_project_on_agent=cache_project_on_agent,
        )
        self.evaluate(self.dt_project_info.id)

    def evaluate(self, pred_project_id):
        self.evaluator = self._get_evaluator_class()(
            api=self.api,
            gt_project_id=self.gt_project_id,
            pred_project_id=pred_project_id,
            result_dir=self.output_dir,
        )
        self.evaluator.evaluate()
