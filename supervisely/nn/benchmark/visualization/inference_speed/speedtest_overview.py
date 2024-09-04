from __future__ import annotations

from typing import TYPE_CHECKING

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class SpeedtestOverview(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.clickable = False
        num_iterations = self._loader.speedtest["speedtest"][0]["num_iterations"]
        self.schema = Schema(
            self._loader.inference_speed_text,
            markdown_speedtest_overview=Widget.Markdown(
                title="Overview", formats=[num_iterations, self._loader.hardware]
            ),
            table=Widget.Table(),
        )
        self._row_ids = None

    def get_table(self, widget: Widget.Table) -> dict:
        res = {}

        columns = [
            "Runtime",
            "Batch size 1 (ms)",
            "Batch size 8 (ms)",
            "Batch size 16 (ms)",
            "Batch size 1 (FPS)",
            "Batch size 8 (FPS)",
            "Batch size 16 (FPS)",
        ]
        temp_res = {}
        for test in self._loader.speedtest["speedtest"]:
            device = "GPU" if "cuda" in test["device"] else "CPU"
            runtime = test["runtime"]
            batch_size = test["batch_size"]
            row_name = f"{device} {runtime}"

            row = temp_res.setdefault(row_name, {col: None for col in columns})
            row["Runtime"] = row_name
            row[f"Batch size {batch_size} (ms)"] = test["benchmark"]["total"]
            row[f"Batch size {batch_size} (FPS)"] = round(
                1000 / test["benchmark"]["total"] * batch_size
            )

        res["content"] = []
        for row in temp_res.values():
            dct = {
                "row": row,
                "id": row["Runtime"],
                "items": list(row.values()),
            }
            res["content"].append(dct)

        columns_options = [
            {"maxWidth": "225px"},
            {"subtitle": "ms", "tooltip": "Milliseconds"},
            {"subtitle": "ms", "tooltip": "Milliseconds"},
            {"subtitle": "ms", "tooltip": "Milliseconds"},
            {"subtitle": "FPS", "tooltip": "Frames per second"},
            {"subtitle": "FPS", "tooltip": "Frames per second"},
            {"subtitle": "FPS", "tooltip": "Frames per second"},
        ]

        res["columns"] = columns
        res["columnsOptions"] = columns_options

        return res

    def get_table_click_data(self, widget: Widget.Table) -> dict:
        return {}
