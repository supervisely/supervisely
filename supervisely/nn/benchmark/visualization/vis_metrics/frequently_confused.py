from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class FrequentlyConfused(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)

        self.clickable: bool = True
        self.switchable: bool = True
        self._keypair_sep: str = " - "
        df = self._loader.mp.frequently_confused()
        if df.empty:
            self.schema = Schema(
                self._loader.vis_texts,
                empty=Widget.Markdown(
                    title="Frequently Confused Classes",
                    is_header=True,
                    formats=[
                        "Frequently Confused Classes",
                        "No frequently confused class pairs found",
                    ],
                ),
            )
            self.empty = True
            return

        pair = df["category_pair"][0]
        prob = df["probability"][0]
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_frequently_confused=Widget.Markdown(
                title="Frequently Confused Classes",
                is_header=True,
                formats=[
                    pair[0],
                    pair[1],
                    prob.round(2),
                    pair[0],
                    pair[1],
                    (prob * 100).round(),
                    pair[0],
                    pair[1],
                    pair[1],
                    pair[0],
                ],
            ),
            chart_01=Widget.Chart(switch_key="probability"),
            chart_02=Widget.Chart(switch_key="count"),
        )

    def get_figure(self, widget: Widget.Chart):  # -> Optional[Tuple[go.Figure]]:
        if self.empty:
            return

        import plotly.graph_objects as go  # pylint: disable=import-error

        # Frequency of confusion as bar chart
        confused_df = self._loader.mp.frequently_confused()
        confused_name_pairs = confused_df["category_pair"]
        x_labels = [f"{pair[0]} - {pair[1]}" for pair in confused_name_pairs]
        y_labels = confused_df[widget.switch_key]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=x_labels, y=y_labels, marker=dict(color=y_labels, colorscale="Reds"))
        )
        fig.update_layout(
            # title="Frequently confused class pairs",
            xaxis_title="Class Pair",
            yaxis_title=y_labels.name.capitalize(),
        )
        fig.update_traces(text=y_labels.round(2))
        fig.update_traces(
            hovertemplate="Class Pair: %{x}<br>"
            + y_labels.name.capitalize()
            + ": %{y:.2f}<extra></extra>"
        )
        return fig

    def get_click_data(self, widget: Widget.Chart) -> Optional[dict]:
        if not self.clickable or self.empty:
            return
        res = dict(projectMeta=self._loader.dt_project_meta.to_json())

        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}

        for keypair, v in self._loader.click_data.frequently_confused.items():
            subkey1, subkey2 = keypair
            key = subkey1 + self._keypair_sep + subkey2
            res["clickData"][key] = {}
            res["clickData"][key]["imagesIds"] = []
            res["clickData"][key]["title"] = f"Confused classes: {subkey1} - {subkey2}"

            img_ids = set()
            obj_ids = set()
            for x in v:
                img_ids.add(x["dt_img_id"])
                obj_ids.add(x["dt_obj_id"])

            res["clickData"][key]["imagesIds"] = list(img_ids)
            res["clickData"][key]["filters"] = [
                {"type": "specific_objects", "tagId": None, "value": list(obj_ids)},
            ]

        return res
