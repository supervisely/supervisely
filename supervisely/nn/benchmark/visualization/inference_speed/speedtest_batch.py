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

        ms_color = "#e377c2"
        fps_color = "#17becf"

        temp_res = {}
        for test in self._loader.speedtest["speedtest"]:
            batch_size = test["batch_size"]

            std = test["benchmark_std"]["total"]
            ms = test["benchmark"]["total"]
            fps = round(1000 / test["benchmark"]["total"] * batch_size)

            ms_line = temp_res.setdefault("ms", {})
            fps_line = temp_res.setdefault("fps", {})

            ms_line[batch_size] = ms
            fps_line[batch_size] = fps

            ms_lower, ms_upper = ms - std, ms + std
            ms_lower_line = temp_res.setdefault("ms_lower", {})
            ms_upper_line = temp_res.setdefault("ms_upper", {})
            ms_lower_line[batch_size] = ms_lower
            ms_upper_line[batch_size] = ms_upper

            fps_lower = round(1000 / ms_lower * batch_size)
            fps_upper = round(1000 / ms_upper * batch_size)
            fps_lower_line = temp_res.setdefault("fps_lower", {})
            fps_upper_line = temp_res.setdefault("fps_upper", {})
            fps_lower_line[batch_size] = fps_lower
            fps_upper_line[batch_size] = fps_upper

        max_y_fps = max(fps_upper_line.values())
        max_y_ms = max(ms_upper_line.values())

        fig = go.Figure()

        # first add the range lines
        fig.add_trace(
            go.Scatter(
                x=list(temp_res["ms_upper"].keys()) + list(temp_res["ms_lower"].keys())[::-1],
                y=list(temp_res["ms_upper"].values()) + list(temp_res["ms_lower"].values())[::-1],
                fill="toself",
                fillcolor="rgba(240, 184, 223, 0.3)",
                line_color="rgba(255,255,255,0)",
                showlegend=False,
                name="Infrence time (ms)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(temp_res["fps_upper"].keys()) + list(temp_res["fps_lower"].keys())[::-1],
                y=list(temp_res["fps_upper"].values()) + list(temp_res["fps_lower"].values())[::-1],
                fill="toself",
                yaxis="y2",
                fillcolor="rgba(184, 241, 247, 0.3)",
                line_color="rgba(255,255,255,0)",
                showlegend=False,
                name="FPS",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(temp_res["ms"].keys()),
                y=list(temp_res["ms"].values()),
                name="Infrence time (ms)",
                line=dict(color=ms_color),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(temp_res["fps"].keys()),
                y=list(temp_res["fps"].values()),
                name="FPS",
                yaxis="y2",
                line=dict(color=fps_color, dash="dot"),
            )
        )
        fig.update_layout(
            yaxis=dict(
                title="Infrence time (ms)",
                range=[0, max_y_ms * 1.1],
            ),
            yaxis2=dict(
                title="FPS (Images per second)",
                overlaying="y",
                side="right",
                range=[0, max_y_fps * 1.1],
            ),
            xaxis=dict(title="Batch Size", dtick=1),
            width=700,
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
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
