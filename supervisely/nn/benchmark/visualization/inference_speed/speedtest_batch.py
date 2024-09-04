from __future__ import annotations

from typing import TYPE_CHECKING

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class SpeedtestBatch(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            self._loader.inference_speed_text,
            markdown_batch_inference=Widget.Markdown(title="Batch inference"),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go  # pylint: disable=import-error
        from plotly.subplots import make_subplots  # pylint: disable=import-error

        ms_colors = iter(["#f0b8df", "#b8f1f7", "#e8e983", "#ffbe84", "#c7b0dd", "#82dd82"])
        fps_colors = iter(["#e377c2", "#17becf", "#bcbd22", "#ff7f0e", "#9467bd", "#2ca02c"])

        temp_res = {}
        for test in self._loader.speedtest["speedtest"]:
            batch_size = test["batch_size"]

            ms = test["benchmark"]["total"]
            fps = round(1000 / test["benchmark"]["total"] * batch_size)

            ms_line = temp_res.setdefault("ms", {})
            fps_line = temp_res.setdefault("fps", {})

            ms_line[batch_size] = ms
            fps_line[batch_size] = fps

        max_y = max(ms_line.values())
        max_y_idx = max(ms_line, key=ms_line.get)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(temp_res["ms"].keys()),
                y=list(temp_res["ms"].values()),
                name="Infrence time (ms)",
                line=dict(color=next(ms_colors)),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(temp_res["fps"].keys()),
                y=list(temp_res["fps"].values()),
                name="FPS",
                yaxis="y2",
                line=dict(color=next(fps_colors), dash="dot"),
            )
        )
        fig.update_layout(
            yaxis=dict(
                title="Infrence time (ms)",
            ),
            yaxis2=dict(
                title="FPS",
                overlaying="y",
                side="right",
            ),
            xaxis_title="Batch Size",
            width=700,
            height=500,
            legend=dict(x=max_y_idx, y=max_y * 0.7),
        )

        return fig


## ========================backup (for public benchmark)==========================
# class SpeedtestBatch(MetricVis):

#     def __init__(self, loader: Visualizer) -> None:
#         super().__init__(loader)
#         self.switchable: bool = True
#         self.schema = Schema(
#             self._loader.inference_speed_text,
#             markdown_batch_inference=Widget.Markdown(title="Batch inference"),
#             chart1=Widget.Chart(switch_key="ms"),
#             chart2=Widget.Chart(switch_key="fps"),
#         )

#     def get_figure(self, widget: Widget.Chart):  #  -> Optional[go.Figure]
#         import plotly.graph_objects as go  # pylint: disable=import-error

#         colors = iter(["#17becf", "#e377c2", "#bcbd22", "#ff7f0e", "#9467bd", "#2ca02c"])

#         data = {}
#         for test in self._loader.speedtest["speedtest"]:
#             device = "GPU" if "cuda" in test["device"] else "CPU"
#             runtime = test["runtime"]
#             runtime_and_device = f"{device} {runtime}"
#             batch_size = test["batch_size"]

#             if widget.switch_key == "ms":
#                 total = test["benchmark"]["total"]
#             else:
#                 total = round(1000 / test["benchmark"]["total"] * batch_size)

#             line = data.setdefault(runtime_and_device, {})
#             line[batch_size] = total

#         fig = go.Figure()
#         min_x, max_y, min_x_idx = float("inf"), 0, 0
#         for idx, (runtime_and_device, line) in enumerate(data.items()):
#             max_y = max(max_y, max(line.values()))
#             if min_x > min(line.keys()):
#                 min_x = min(line.keys())
#                 min_x_idx = idx
#             fig.add_trace(
#                 go.Scatter(
#                     x=list(line.keys()),
#                     y=list(line.values()),
#                     mode="lines+markers",
#                     name=runtime_and_device,
#                     line=dict(color=next(colors)),
#                 )
#             )

#         fig.update_layout(
#             xaxis_title="Batch Size",
#             yaxis_title="Time (ms)" if widget.switch_key == "ms" else "Images per second (FPS)",
#             legend=dict(x=min_x_idx, y=max_y * 0.7),
#             width=700,
#             height=500,
#         )

#         if widget.switch_key == "ms":
#             hovertemplate = "Batch Size: %{x}<br>Time: %{y:.2f} ms<extra></extra>"
#         else:
#             hovertemplate = "Batch Size: %{x}<br>FPS: %{y:.2f}<extra></extra>"
#         fig.update_traces(hovertemplate=hovertemplate)
#         return fig
