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

        first_cmat, _ = self.eval_results[0].mp.confusion_matrix
        n_classes = int(first_cmat.shape[0])

        classes = list(self.eval_results[0].classes_whitelist)
        if len(classes) < n_classes:
            classes = classes + [str(i) for i in range(len(classes), n_classes)]
        elif len(classes) > n_classes:
            classes = classes[:n_classes]

        model_cnt = len(self.eval_results)
        all_models_cmat = np.zeros((model_cnt, n_classes, n_classes))
        for model_idx, eval_result in enumerate(self.eval_results):
            cmat, _ = eval_result.mp.confusion_matrix
            if cmat.shape != (n_classes, n_classes):
                tmp = np.zeros((n_classes, n_classes))
                r = min(n_classes, cmat.shape[0])
                c = min(n_classes, cmat.shape[1])
                tmp[:r, :c] = cmat[:r, :c]
                cmat = tmp
            all_models_cmat[model_idx] = cmat[::-1].copy()

        sum_cmat = all_models_cmat.sum(axis=0)
        np.fill_diagonal(sum_cmat, 0)
        sum_cmat_flat = sum_cmat.flatten()
        sorted_indices = np.argsort(sum_cmat_flat)[::-1]
        n_pairs = min(10, n_classes * (n_classes - 1))
        sorted_indices = sorted_indices[:n_pairs]
        rows = sorted_indices // n_classes
        cols = sorted_indices % n_classes
        labels = [f"{classes[rows[i]]}-{classes[cols[i]]}" for i in range(n_pairs)]
        for model_idx, eval_result in enumerate(self.eval_results):
            cmat = all_models_cmat[model_idx]
            probs = cmat[rows, cols] 
            probs = probs * 100
            fig.add_trace(
                go.Bar(
                    name=eval_result.name,
                    x=labels,
                    y=probs,
                    hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                    marker=dict(color=eval_result.color, line=dict(width=0.7)),
                )
            )

        return fig
