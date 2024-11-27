from typing import Dict

from supervisely.nn.benchmark.semantic_segmentation.base_vis_metric import (
    SemanticSegmVisMetric,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class FrequentlyConfused(SemanticSegmVisMetric):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clickable = True
        self._keypair_sep = "-"

    @property
    def md(self) -> MarkdownWidget:
        if self.is_empty:
            text = self.vis_texts.markdown_frequently_confused_empty
        else:
            text = self.vis_texts.markdown_frequently_confused
        return MarkdownWidget("frequently_confused", "Frequently Confused Classes", text=text)

    @property
    def chart(self) -> ChartWidget:
        chart = ChartWidget("frequently_confused", self.get_figure())
        chart.set_click_data(
            self.explore_modal_table.id,
            self.get_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].x}`, 'keySeparator': '-',",
        )
        return chart

    @property
    def is_empty(self) -> bool:
        probs, indexes_2d = self.eval_result.mp.frequently_confused
        return len(probs) == 0

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        # Frequency of Confused Classes figure
        probs, indexes_2d = self.eval_result.mp.frequently_confused
        confused_classes = []
        for idx in indexes_2d:
            gt_idx, pred_idx = idx[0], idx[1]
            gt_class = self.eval_result.mp.eval_data.index[gt_idx]
            pred_class = self.eval_result.mp.eval_data.index[pred_idx]
            confused_classes.append(f"{gt_class}-{pred_class}")

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=confused_classes,
                y=probs,
                orientation="v",
                text=probs,
                marker=dict(color=probs, colorscale="Reds"),
            )
        )
        fig.update_traces(hovertemplate="Class Pair: %{x}<br>Probability: %{y:.2f}<extra></extra>")
        fig.update_layout(
            xaxis_title="Class Pair",
            yaxis_title="Probability",
            yaxis_range=[0, max(probs) + 0.1],
            yaxis=dict(showticklabels=False),
            font=dict(size=24),
            width=1000 if len(confused_classes) > 10 else 600,
        )

        return fig

    def get_click_data(self) -> Dict:
        if self.is_empty:
            return
        res = dict(projectMeta=self.eval_result.pred_project_meta.to_json())

        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}

        _, class_names = self.eval_result.mp.confusion_matrix
        _, indexes_2d = self.eval_result.mp.frequently_confused
        for idx in indexes_2d:
            gt_idx, pred_idx = idx[0], idx[1]
            gt_key = class_names[gt_idx]
            pred_key = class_names[pred_idx]
            key = f"{gt_key}{self._keypair_sep}{pred_key}"

            res["clickData"][key] = {}
            res["clickData"][key]["imagesIds"] = []
            idx_key = str(gt_idx) + "_" + str(pred_idx)
            for name in self.eval_result.mp.cmat_cell_img_names[idx_key]:
                gt_img_id = self.eval_result.images_map[name]
                pred_img_id = self.eval_result.matched_pair_data[gt_img_id].pred_image_info.id
                res["clickData"][key]["imagesIds"].append(pred_img_id)

            title = f"Confused classes. GT: '{gt_key}' ― Predicted: '{pred_key}'"
            res["clickData"][key]["title"] = title
        return res
