from typing import Dict

from supervisely.nn.benchmark.semantic_segmentation.base_vis_metric import (
    SemanticSegmVisMetric,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class ConfusionMatrix(SemanticSegmVisMetric):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clickable = True
        self._keypair_sep = "-"

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            "confusion_matrix",
            "Confusion Matrix",
            text=self.vis_texts.markdown_confusion_matrix,
        )

    @property
    def chart(self) -> ChartWidget:
        chart = ChartWidget("confusion_matrix", self.get_figure())
        chart.set_click_data(
            self.explore_modal_table.id,
            self.get_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].y}${'-'}${payload.points[0].x}`, 'keySeparator': '-',",
        )
        return chart

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        # # Confusion Matrix figure
        confusion_matrix, class_names = self.eval_result.mp.confusion_matrix

        x = [el for el in class_names if el != "mean"]
        y = x[::-1].copy()
        if len(x) >= 20:
            text_anns = [[str(el) for el in row] for row in confusion_matrix]
        else:
            text_anns = [
                [
                    f"Predicted: {pred}<br>Ground Truth: {gt}<br> Probability: {confusion_matrix[ig][ip]}"
                    for ip, pred in enumerate(x)
                ]
                for ig, gt in enumerate(y)
            ]

        fig.add_trace(
            go.Heatmap(
                z=confusion_matrix,
                x=x,
                y=y,
                colorscale="Viridis",
                showscale=False,
                text=text_anns,
                hoverinfo="text",
            )
        )

        fig.update_layout(xaxis_title="Predicted", yaxis_title="Ground Truth")
        if len(x) <= 20:
            fig.update_layout(width=600, height=600)
        return fig

    def get_click_data(self) -> Dict:
        res = dict(projectMeta=self.eval_result.pred_project_meta.to_json())
        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}

        _, class_names = self.eval_result.mp.confusion_matrix
        for ig, gt_key in enumerate(class_names):
            for ip, pred_key in enumerate(class_names):
                key = f"{gt_key}{self._keypair_sep}{pred_key}"
                res["clickData"][key] = {}
                res["clickData"][key]["imagesIds"] = []

                cmat_key = str(ig) + "_" + str(ip)
                for name in self.eval_result.mp.cmat_cell_img_names[cmat_key]:
                    gt_img_id = self.eval_result.images_map[name]
                    pred_img_id = self.eval_result.matched_pair_data[gt_img_id].pred_image_info.id
                    res["clickData"][key]["imagesIds"].append(pred_img_id)
                title = f"Confusion Matrix. GT: '{gt_key}' ― Predicted: '{pred_key}'"
                res["clickData"][key]["title"] = title

        return res
