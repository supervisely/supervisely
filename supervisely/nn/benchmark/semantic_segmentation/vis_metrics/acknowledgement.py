from supervisely.nn.benchmark.semantic_segmentation.base_vis_metric import (
    SemanticSegmVisMetric,
)
from supervisely.nn.benchmark.visualization.widgets import MarkdownWidget


class Acknowledgement(SemanticSegmVisMetric):

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            "acknowledgement",
            "Acknowledgement",
            text=self.vis_texts.markdown_acknowledgement,
        )
