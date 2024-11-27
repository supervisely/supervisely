from typing import List, Union

from supervisely.nn.benchmark.semantic_segmentation.base_vis_metric import (
    SemanticSegmVisMetric,
)
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    MarkdownWidget,
    TableWidget,
)


class Speedtest(SemanticSegmVisMetric):

    def is_empty(self) -> bool:
        return self.eval_result.speedtest_info is None

    def multiple_batche_sizes(self) -> bool:
        return len(self.eval_result.speedtest_info["speedtest"]) > 1

    @property
    def latency(self) -> List[Union[int, str]]:
        if self.eval_result.speedtest_info is None:
            return ["N/A"]
        latency = []
        for test in self.eval_result.speedtest_info["speedtest"]:
            latency.append(round(test["benchmark"]["total"], 2))
        return latency

    @property
    def fps(self) -> List[Union[int, str]]:
        if self.eval_result.speedtest_info is None:
            return ["N/A"]
        fps = []
        for test in self.eval_result.speedtest_info["speedtest"]:
            fps.append(round(1000 / test["benchmark"]["total"], 2))
        return fps

    @property
    def intro_md(self) -> MarkdownWidget:
        device = self.eval_result.speedtest_info["model_info"]["device"]
        hardware = self.eval_result.speedtest_info["model_info"]["hardware"]
        runtime = self.eval_result.speedtest_info["model_info"]["runtime"]
        num_it = self.eval_result.speedtest_info["speedtest"][0]["num_iterations"]

        return MarkdownWidget(
            name="speedtest_intro",
            title="Inference Speed",
            text=self.vis_texts.markdown_speedtest_intro.format(device, hardware, runtime, num_it),
        )

    @property
    def intro_table(self) -> TableWidget:
        res = {}

        columns = [" ", "Inference time", "FPS"]
        temp_res = {}
        max_fps = 0
        for test in self.eval_result.speedtest_info["speedtest"]:
            batch_size = test["batch_size"]

            ms = round(test["benchmark"]["total"], 2)
            fps = round(1000 / test["benchmark"]["total"] * batch_size)
            row = [batch_size, ms, fps]
            temp_res[batch_size] = row
            max_fps = max(max_fps, fps)

        res["content"] = []
        # sort by batch size
        temp_res = dict(sorted(temp_res.items()))
        for row in temp_res.values():
            dct = {
                "row": row,
                "id": row[0],
                "items": row,
            }
            res["content"].append(dct)

        columns_options = [
            {"disableSort": True},  # "customCell": True
            {"subtitle": "ms", "tooltip": "Milliseconds for batch images", "postfix": "ms"},
            {
                "subtitle": "imgs/sec",
                "tooltip": "Frames (images) per second",
                "postfix": "fps",
                "maxValue": max_fps,
            },
        ]

        res["columns"] = columns
        res["columnsOptions"] = columns_options

        table = TableWidget(
            name="speedtest_intro_table",
            data=res,
            show_header_controls=False,
            fix_columns=1,
        )
        # table.main_column = "Batch size"
        table.fixed_columns = 1
        table.show_header_controls = False
        return table

    @property
    def batch_size_md(self) -> MarkdownWidget:
        return MarkdownWidget(
            name="batch_size",
            title="Batch Size",
            text=self.vis_texts.markdown_batch_inference,
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(name="speed_charts", figure=self.get_figure())

    def get_figure(self):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go  # pylint: disable=import-error
        from plotly.subplots import make_subplots  # pylint: disable=import-error

        fig = make_subplots(cols=2)

        ms_color = "#e377c2"
        fps_color = "#17becf"

        temp_res = {}
        for test in self.eval_result.speedtest_info["speedtest"]:
            batch_size = test["batch_size"]

            std = test["benchmark_std"]["total"]
            ms = test["benchmark"]["total"]
            fps = round(1000 / test["benchmark"]["total"] * batch_size)

            ms_line = temp_res.setdefault("ms", {})
            fps_line = temp_res.setdefault("fps", {})
            ms_std_line = temp_res.setdefault("ms_std", {})

            ms_line[batch_size] = ms
            fps_line[batch_size] = fps
            ms_std_line[batch_size] = round(std, 2)

        fig.add_trace(
            go.Scatter(
                x=list(temp_res["ms"].keys()),
                y=list(temp_res["ms"].values()),
                name="Inference time (ms)",
                line=dict(color=ms_color),
                customdata=list(temp_res["ms_std"].values()),
                error_y=dict(
                    type="data",
                    array=list(temp_res["ms_std"].values()),
                    visible=True,
                    color="rgba(227, 119, 194, 0.7)",
                ),
                hovertemplate="Batch Size: %{x}<br>Time: %{y:.2f} ms<br> Standard deviation: %{customdata:.2f} ms<extra></extra>",
            ),
            col=1,
            row=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(temp_res["fps"].keys()),
                y=list(temp_res["fps"].values()),
                name="FPS",
                line=dict(color=fps_color),
                hovertemplate="Batch Size: %{x}<br>FPS: %{y:.2f}<extra></extra>",
            ),
            col=2,
            row=1,
        )

        fig.update_xaxes(title_text="Batch size", col=1, dtick=1)
        fig.update_xaxes(title_text="Batch size", col=2, dtick=1)

        fig.update_yaxes(title_text="Time (ms)", col=1)
        fig.update_yaxes(title_text="FPS", col=2)
        fig.update_layout(height=400)

        return fig
