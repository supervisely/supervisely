from typing import List

from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class FrequentlyConfused(BaseVisMetrics):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eval_results: List[SemanticSegmentationEvalResult]
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
        return ChartWidget("frequently_confused", self.get_figure())

    @property
    def is_empty(self) -> bool:
        return all(len(e.mp.frequently_confused[0]) == 0 for e in self.eval_results)

    def get_figure(self):
        import numpy as np
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        classes = self.eval_results[0].classes_whitelist

        model_cnt = len(self.eval_results)
        all_models_cmat = np.zeros((model_cnt, len(classes), len(classes)))
        for model_idx, eval_result in enumerate(self.eval_results):
            cmat, _ = eval_result.mp.confusion_matrix
            all_models_cmat[model_idx] = cmat.copy()

        sum_cmat = all_models_cmat.sum(axis=0)
        np.fill_diagonal(sum_cmat, 0)
        sum_cmat_flat = sum_cmat.flatten()
        sorted_indices = np.argsort(sum_cmat_flat)[::-1]
        n_pairs = 10
        sorted_indices = sorted_indices[:n_pairs]
        rows = sorted_indices // len(classes)
        cols = sorted_indices % len(classes)
        for model_idx, eval_result in enumerate(self.eval_results):
            cmat = all_models_cmat[model_idx]
            probs, indexes_2d = eval_result.mp.get_frequently_confused(cmat, n_pairs=n_pairs)
            probs = probs * 100
            labels = [
                f"{classes[rows[i]]}{self._keypair_sep}{classes[cols[i]]}" for i in range(n_pairs)
            ]
            fig.add_trace(
                go.Bar(
                    name=eval_result.name,
                    x=labels,
                    y=probs,
                    hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                    marker=dict(color=eval_result.color),
                )
            )

        return fig