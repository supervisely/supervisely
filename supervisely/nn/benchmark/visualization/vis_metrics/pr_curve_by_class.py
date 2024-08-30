from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class PRCurveByClass(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.clickable = True
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_pr_by_class=Widget.Markdown(title="PR Curve by Class"),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart):  # -> Optional[go.Figure]:
        import plotly.express as px  # pylint: disable=import-error

        # Precision-Recall curve per-class
        df = pd.DataFrame(self._loader.mp.pr_curve(), columns=self._loader.mp.cat_names)

        fig = px.line(
            df,
            x=self._loader.mp.recThrs,
            y=df.columns,
            # title="Precision-Recall Curve per Class",
            labels={"x": "Recall", "value": "Precision", "variable": "Category"},
            color_discrete_sequence=px.colors.qualitative.Prism,
            width=800,
            height=600,
        )

        fig.update_yaxes(range=[0, 1])
        fig.update_xaxes(range=[0, 1])
        # fig.show()

        return fig
