from __future__ import annotations

from typing import TYPE_CHECKING

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class SpeedtestRealTime(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.switchable: bool = True
        charts = {}
        for batch_size in [1, 8, 16]:
            for measure in ["ms", "fps"]:
                key = f"{batch_size}_{measure}"
                charts[f"chart_{key}"] = Widget.Chart(switch_key=key)
        self.schema = Schema(
            self._loader.inference_speed_text,
            markdown_real_time_inference=Widget.Markdown(title="Real-time inference"),
            **charts,
        )

    def get_figure(self, widget: Widget.Chart):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go  # pylint: disable=import-error

        colors = iter(["#17becf", "#e377c2", "#bcbd22", "#ff7f0e", "#9467bd", "#2ca02c"])

        batch_size, measure = widget.switch_key.split("_")

        fig = go.Figure()
        for test in self._loader.speedtest["speedtest"]:
            device = "GPU" if "cuda" in test["device"] else "CPU"
            runtime = test["runtime"]
            runtime_and_device = f"{device} {runtime}"
            bs = test["batch_size"]
            if batch_size != bs:
                continue

            if measure == "ms":
                total = test["benchmark"]["total"]
            else:
                total = round(1000 / test["benchmark"]["total"] * bs)

            fig.add_trace(
                go.Bar(
                    y=total,
                    x=runtime_and_device,
                    marker=dict(color=next(colors)),
                )
            )


        y_title = "Time (ms)" if measure == "ms" else "Images per second (FPS)"
        fig.update_xaxes(title_text="Runtime")
        fig.update_yaxes(title_text=y_title)
        fig.update_layout(height=400, width=800)

        return fig
