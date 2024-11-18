from supervisely.nn.benchmark.semantic_segmentation.base_vis_metric import (
    SemanticSegmVisMetric,
)
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import MarkdownWidget


class Acknowledgement(SemanticSegmVisMetric):
    def __init__(self, vis_texts, eval_result: SemanticSegmentationEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            "acknowledgement",
            "Acknowledgement",
            text=self.vis_texts.markdown_acknowledgement,
        )
