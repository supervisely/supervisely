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
                title="Overview", formats=[num_iterations]
            ),
            table=Widget.Table(),
        )
        self._row_ids = None

    def get_table(self, widget: Widget.Table) -> dict:
        res = {}

        columns = ["Batch size", "Infrence time", "FPS"]
        temp_res = {}
        max_fps = 0
        for test in self._loader.speedtest["speedtest"]:
            batch_size = test["batch_size"]

            row = temp_res.setdefault(batch_size, {})
            row["Batch size"] = batch_size
            row["Inference time"] = round(test["benchmark"]["total"], 2)
            fps = round(1000 / test["benchmark"]["total"] * batch_size)
            max_fps = max(max_fps, fps)
            row["FPS"] = fps

        res["content"] = []
        for row in temp_res.values():
            dct = {
                "row": row,
                "id": row["Batch size"],
                "items": list(row.values()),
            }
            res["content"].append(dct)

        columns_options = [
            {"maxWidth": "225px", "postfix": "images"},
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

        return res

    def get_table_click_data(self, widget: Widget.Table) -> dict:
        return {}


## ========================backup (for public benchmark)==========================
# class SpeedtestOverview(MetricVis):

#     def __init__(self, loader: Visualizer) -> None:
#         super().__init__(loader)
#         self.clickable = False
#         num_iterations = self._loader.speedtest["speedtest"][0]["num_iterations"]
#         self.schema = Schema(
#             self._loader.inference_speed_text,
#             markdown_speedtest_overview=Widget.Markdown(
#                 title="Overview", formats=[num_iterations, self._loader.hardware]
#             ),
#             table=Widget.Table(),
#         )
#         self._row_ids = None

#     def get_table(self, widget: Widget.Table) -> dict:
#         res = {}

#         columns = [
#             "Runtime",
#             "Batch size 1",
#             "Batch size 8",
#             "Batch size 16",
#             "Batch size 1",
#             "Batch size 8",
#             "Batch size 16",
#         ]
#         temp_res = {}
#         for test in self._loader.speedtest["speedtest"]:
#             device = "GPU" if "cuda" in test["device"] else "CPU"
#             runtime = test["runtime"]
#             batch_size = test["batch_size"]
#             row_name = f"{device} {runtime}"

#             row = temp_res.setdefault(row_name, {})
#             row["Runtime"] = row_name
#             row[f"Batch size {batch_size} (ms)"] = round(test["benchmark"]["total"], 2)
#             row[f"Batch size {batch_size} (FPS)"] = round(
#                 1000 / test["benchmark"]["total"] * batch_size
#             )

#         res["content"] = []
#         for row in temp_res.values():
#             dct = {
#                 "row": row,
#                 "id": row["Runtime"],
#                 "items": list(row.values()),
#             }
#             res["content"].append(dct)

#         columns_options = [
#             {"maxWidth": "225px"},
#             {"subtitle": "ms", "tooltip": "Milliseconds for 1 image"},
#             {"subtitle": "ms", "tooltip": "Milliseconds for 8 images"},
#             {"subtitle": "ms", "tooltip": "Milliseconds for 16 images"},
#             {"subtitle": "FPS", "tooltip": "Frames (images) per second"},
#             {"subtitle": "FPS", "tooltip": "Frames (images) per second"},
#             {"subtitle": "FPS", "tooltip": "Frames (images) per second"},
#         ]

#         res["columns"] = columns
#         res["columnsOptions"] = columns_options

#         return res

#     def get_table_click_data(self, widget: Widget.Table) -> dict:
#         return {}
