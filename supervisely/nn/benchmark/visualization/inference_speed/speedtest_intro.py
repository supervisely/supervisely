from __future__ import annotations

from typing import TYPE_CHECKING

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class SpeedtestIntro(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        speedtest = self._loader.speedtest["speedtest"][0]
        self.schema = Schema(
            self._loader.inference_speed_text,
            markdown_speedtest_intro=Widget.Markdown(
                title="Inference Speed",
                is_header=True,
                formats=[
                    speedtest["device"],
                    self._loader.hardware,
                    speedtest["runtime"]
                ],
            ),
        )
