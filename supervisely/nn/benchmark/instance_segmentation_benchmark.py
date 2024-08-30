from supervisely.nn.benchmark.base_benchmark import BaseBenchmark
from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.nn.benchmark.evaluation import InstanceSegmentationEvaluator
from supervisely.nn.benchmark.utils import try_set_conf_auto

CONF_THRES = 0.05


class InstanceSegmentationBenchmark(BaseBenchmark):

    @property
    def cv_task(self) -> str:
        return CVTask.INSTANCE_SEGMENTATION

    def _get_evaluator_class(self) -> type:
        return InstanceSegmentationEvaluator

    def _run_inference(
        self,
        output_project_id=None,
        batch_size: int = 8,
        cache_project_on_agent: bool = False,
    ):
        assert try_set_conf_auto(
            self.session, CONF_THRES
        ), f"Unable to set the confidence threshold to {CONF_THRES} for evalation."
        return super()._run_inference(output_project_id, batch_size, cache_project_on_agent)
