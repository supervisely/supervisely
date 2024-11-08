from __future__ import annotations

from typing import Dict, Literal

from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.object_detection.evaluator import (
    ObjectDetectionEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class FrequentlyConfused(BaseVisMetric):
    MARKDOWN = "frequently_confused"
    MARKDOWN_EMPTY = "frequently_confused_empty"
    CHART = "frequently_confused"

    def __init__(self, vis_texts, eval_result: ObjectDetectionEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result
        self.clickable = True
        self.df = self.eval_result.mp.frequently_confused()
        self._keypair_sep: str = " - "
        self.is_empty = self.df.empty
        self.switchable = True

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_frequently_confused
        pair = self.df["category_pair"][0]
        prob = self.df["probability"][0]
        text = text.format(
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
        )
        return MarkdownWidget(self.MARKDOWN, "Frequently Confused Classes", text)

    @property
    def chart_probability(self) -> ChartWidget:
        return ChartWidget(self.CHART, self._get_figure("probability"))

    @property
    def chart_count(self) -> ChartWidget:
        return ChartWidget(self.CHART, self._get_figure("count"))

    @property
    def empty_md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_frequently_confused_empty
        return MarkdownWidget(self.MARKDOWN_EMPTY, "Frequently Confused Classes", text)

    def _get_figure(self, switch_key: Literal["probability", "count"]):  # -> go.Figure:
        if self.empty:
            return

        import plotly.graph_objects as go  # pylint: disable=import-error

        # Frequency of confusion as bar chart
        confused_df = self.eval_result.mp.frequently_confused()
        confused_name_pairs = confused_df["category_pair"]
        x_labels = [f"{pair[0]} - {pair[1]}" for pair in confused_name_pairs]
        y_labels = confused_df[switch_key]

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

    def get_click_data(self) -> Dict:
        if not self.clickable or self.empty:
            return
        res = dict(projectMeta=self.eval_result.pred_project_meta.to_json())

        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}

        for keypair, v in self.eval_result.click_data.frequently_confused.items():
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
