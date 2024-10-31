from __future__ import annotations

from typing import TYPE_CHECKING

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class WhatIs(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_what_is=Widget.Markdown(title="What is YOLOv8 model", is_header=True),
            markdown_experts=Widget.Markdown(title="Expert Insights", is_header=True),
            markdown_how_to_use=Widget.Markdown(
                title="How To Use: Training, Inference, Evaluation Loop", is_header=True
            ),
        )
