from typing import List, Union

from supervisely.imaging.color import hex2rgb
from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    MarkdownWidget,
    TableWidget,
)


class Speedtest(BaseVisMetrics):

    def is_empty(self) -> bool:
        return not any(eval_result.speedtest_info for eval_result in self.eval_results)

    def multiple_batche_sizes(self) -> bool:
        for eval_result in self.eval_results:
            if eval_result.speedtest_info is None:
                continue
            if len(eval_result.speedtest_info["speedtest"]) > 1:
                return True
        return False

    @property
    def latency(self) -> List[Union[int, str]]:
        latency = []
        for eval_result in self.eval_results:
            if eval_result.speedtest_info is None:
                latency.append("N/A")
            else:
                added = False
                for test in eval_result.speedtest_info["speedtest"]:
                    if test["batch_size"] == 1:
                        latency.append(round(test["benchmark"]["total"], 2))
                        added = True
                        break
                if not added:
                    latency.append("N/A")
        return latency

    @property
    def fps(self) -> List[Union[int, str]]:
        fps = []
        for eval_result in self.eval_results:
            if eval_result.speedtest_info is None:
                fps.append("N/A")
            else:
                added = False
                for test in eval_result.speedtest_info["speedtest"]:
                    if test["batch_size"] == 1:
                        fps.append(round(1000 / test["benchmark"]["total"], 2))
                        added = True
                        break
                if not added:
                    fps.append("N/A")
        return fps

    @property
    def md_intro(self) -> MarkdownWidget:
        return MarkdownWidget(
            name="speedtest_intro",
            title="Inference Speed",
            text=self.vis_texts.markdown_speedtest_intro,
        )

    @property
    def intro_table(self) -> TableWidget:
        columns = ["Model", "Device", "Hardware", "Runtime"]
        columns_options = [{"disableSort": True} for _ in columns]
        content = []
        for i, eval_result in enumerate(self.eval_results, 1):
            name = f"[{i}] {eval_result.name}"
            if eval_result.speedtest_info is None:
                row = [name, "N/A", "N/A", "N/A"]
                dct = {
                    "row": row,
                    "id": name,
                    "items": row,
                }
                content.append(dct)
                continue
            model_info = eval_result.speedtest_info.get("model_info", {})
            device = model_info.get("device", "N/A")
            hardware = model_info.get("hardware", "N/A")
            runtime = model_info.get("runtime", "N/A")
            row = [name, device, hardware, runtime]
            dct = {
                "row": row,
                "id": name,
                "items": row,
            }
            content.append(dct)

        data = {
            "columns": columns,
            "columnsOptions": columns_options,
            "content": content,
        }
        return TableWidget(
            name="speedtest_intro_table",
            data=data,
            show_header_controls=False,
            fix_columns=1,
        )

    @property
    def inference_time_md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_speedtest_overview_ms.format(100)
        return MarkdownWidget(
            name="inference_time_md",
            title="Overview",
            text=text,
        )

    @property
    def fps_md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_speedtest_overview_fps.format(100)
        return MarkdownWidget(
            name="fps_md",
            title="FPS Table",
            text=text,
        )

    @property
    def fps_table(self) -> TableWidget:
        data = {}
        batch_sizes = set()
        max_fps = 0
        for i, eval_result in enumerate(self.eval_results, 1):
            data[i] = {}
            if eval_result.speedtest_info is None:
                continue
            speedtests = eval_result.speedtest_info["speedtest"]
            for test in speedtests:
                batch_size = test["batch_size"]
                fps = round(1000 / test["benchmark"]["total"] * batch_size)
                batch_sizes.add(batch_size)
                max_fps = max(max_fps, fps)
                data[i][batch_size] = fps

        batch_sizes = sorted(batch_sizes)
        columns = ["Model"]
        columns_options = [{"disableSort": True}]
        for batch_size in batch_sizes:
            columns.append(f"Batch size {batch_size}")
            columns_options.append(
                {
                    "subtitle": "imgs/sec",
                    "tooltip": "Frames (images) per second",
                    "postfix": "fps",
                    # "maxValue": max_fps,
                }
            )

        content = []
        for i, eval_result in enumerate(self.eval_results, 1):
            name = f"[{i}] {eval_result.name}"
            row = [name]
            for batch_size in batch_sizes:
                if batch_size in data[i]:
                    row.append(data[i][batch_size])
                else:
                    row.append("―")
            content.append(
                {
                    "row": row,
                    "id": name,
                    "items": row,
                }
            )
        data = {
            "columns": columns,
            "columnsOptions": columns_options,
            "content": content,
        }
        return TableWidget(
            name="fps_table",
            data=data,
            show_header_controls=False,
            fix_columns=1,
        )

    @property
    def inference_time_table(self) -> TableWidget:
        data = {}
        batch_sizes = set()
        for i, eval_result in enumerate(self.eval_results, 1):
            data[i] = {}
            if eval_result.speedtest_info is None:
                continue
            speedtests = eval_result.speedtest_info["speedtest"]
            for test in speedtests:
                batch_size = test["batch_size"]
                ms = round(test["benchmark"]["total"], 2)
                batch_sizes.add(batch_size)
                data[i][batch_size] = ms

        batch_sizes = sorted(batch_sizes)
        columns = ["Model"]
        columns_options = [{"disableSort": True}]
        for batch_size in batch_sizes:
            columns.extend([f"Batch size {batch_size}"])
            columns_options.extend(
                [
                    {"subtitle": "ms", "tooltip": "Milliseconds for batch images", "postfix": "ms"},
                ]
            )

        content = []
        for i, eval_result in enumerate(self.eval_results, 1):
            name = f"[{i}] {eval_result.name}"
            row = [name]
            for batch_size in batch_sizes:
                if batch_size in data[i]:
                    row.append(data[i][batch_size])
                else:
                    row.append("―")
            content.append(
                {
                    "row": row,
                    "id": name,
                    "items": row,
                }
            )

        data = {
            "columns": columns,
            "columnsOptions": columns_options,
            "content": content,
        }
        return TableWidget(
            name="inference_time_md",
            data=data,
            show_header_controls=False,
            fix_columns=1,
        )

    @property
    def batch_inference_md(self):
        return MarkdownWidget(
            name="batch_inference",
            title="Batch Inference",
            text=self.vis_texts.markdown_batch_inference,
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(name="speed_charts", figure=self.get_figure())

    def get_figure(self):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go  # pylint: disable=import-error
        from plotly.subplots import make_subplots  # pylint: disable=import-error

        fig = make_subplots(cols=2)

        for idx, eval_result in enumerate(self.eval_results, 1):
            if eval_result.speedtest_info is None:
                continue
            temp_res = {}
            for test in eval_result.speedtest_info["speedtest"]:
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

            error_color = "rgba(" + ",".join(map(str, hex2rgb(eval_result.color))) + ", 0.5)"
            fig.add_trace(
                go.Scatter(
                    x=list(temp_res["ms"].keys()),
                    y=list(temp_res["ms"].values()),
                    name=f"[{idx}] {eval_result.name} (ms)",
                    line=dict(color=eval_result.color),
                    customdata=list(temp_res["ms_std"].values()),
                    error_y=dict(
                        type="data",
                        array=list(temp_res["ms_std"].values()),
                        visible=True,
                        color=error_color,
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
                    name=f"[{idx}] {eval_result.name} (fps)",
                    line=dict(color=eval_result.color),
                    hovertemplate="Batch Size: %{x}<br>FPS: %{y:.2f}<extra></extra>",  # <br> Standard deviation: %{customdata:.2f}<extra></extra>",
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
