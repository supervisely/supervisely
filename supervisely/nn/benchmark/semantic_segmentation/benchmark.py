from supervisely.nn.benchmark.base_benchmark import BaseBenchmark
from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvaluator,
)
from supervisely.nn.benchmark.semantic_segmentation.visualizer import (
    SemanticSegmentationVisualizer,
)


class SemanticSegmentationBenchmark(BaseBenchmark):
    visualizer_cls = SemanticSegmentationVisualizer

    @property
    def cv_task(self) -> str:
        return CVTask.SEMANTIC_SEGMENTATION

    def _get_evaluator_class(self) -> type:
        return SemanticSegmentationEvaluator

    def _evaluate(self, gt_project_path, pred_project_path):
        eval_results_dir = self.get_eval_results_dir()
        self.evaluator = self._get_evaluator_class()(
            gt_project_path=gt_project_path,
            pred_project_path=pred_project_path,
            result_dir=eval_results_dir,
            progress=self.pbar,
            items_count=self.dt_project_info.items_count,
            classes_whitelist=self.classes_whitelist,
            evaluation_params=self.evaluation_params,
        )
        self.evaluator.evaluate()
