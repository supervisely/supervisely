from __future__ import annotations

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    MarkdownWidget,
    TableWidget,
)


class Speedtest(DetectionVisMetric):
    MARKDOWN_INTRO = "speedtest_intro"
    TABLE_MARKDOWN = "speedtest_table"
    TABLE = "speedtest"
    CHART_MARKDOWN = "speedtest_chart"
    CHART = "speedtest"

    def is_empty(self) -> bool:
        return not self.eval_result.speedtest_info

    @property
    def num_batche_sizes(self) -> int:
        if self.is_empty():
            return 0
        return len(self.eval_result.speedtest_info.get("speedtest", []))

    @property
    def intro_md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_speedtest_intro
        text = text.format(
            self.eval_result.speedtest_info["model_info"]["device"],
            self.eval_result.speedtest_info["model_info"]["hardware"],
            self.eval_result.speedtest_info["model_info"]["runtime"],
        )
        return MarkdownWidget(name=self.MARKDOWN_INTRO, title="Inference Speed", text=text)

    @property
    def table_md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_speedtest_table
        text = text.format(self.eval_result.speedtest_info["speedtest"][0]["num_iterations"])
        return MarkdownWidget(name=self.TABLE_MARKDOWN, title="Speedtest Table", text=text)

    @property
    def table(self) -> TableWidget:
        columns = [" ", "Inference time", "FPS"]
        content = []
        temp_res = {}
        max_fps = 0
        for test in self.eval_result.speedtest_info["speedtest"]:
            batch_size = test["batch_size"]

            ms = round(test["benchmark"]["total"], 2)
            fps = round(1000 / test["benchmark"]["total"] * batch_size)
            row = [f"Batch size {batch_size}", ms, fps]
            temp_res[batch_size] = row
            max_fps = max(max_fps, fps)

        # sort by batch size
        temp_res = dict(sorted(temp_res.items()))
        for row in temp_res.values():
            content.append({"row": row, "id": row[0], "items": row})

        columns_options = [
            {"disableSort": True},  # , "ustomCell": True},
            {"subtitle": "ms", "tooltip": "Milliseconds for batch images", "postfix": "ms"},
            {
                "subtitle": "imgs/sec",
                "tooltip": "Frames (images) per second",
                "postfix": "fps",
                "maxValue": max_fps,
            },
        ]
        data = {"columns": columns, "columnsOptions": columns_options, "content": content}
        table = TableWidget(name=self.TABLE, data=data)
        table.main_column = "Batch size"
        table.fixed_columns = 1
        table.show_header_controls = False
        return table

    @property
    def chart_md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_speedtest_chart
        return MarkdownWidget(name=self.CHART_MARKDOWN, title="Speedtest Chart", text=text)

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(name=self.CHART, figure=self._get_figure())

    def _get_figure(self):  #  -> go.Figure:
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
