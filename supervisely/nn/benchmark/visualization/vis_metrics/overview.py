from __future__ import annotations

from typing import TYPE_CHECKING

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class Overview(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        info = loader.inference_info
        url = info.get("checkpoint_url")
        link_text = info.get("custom_checkpoint_path")
        if link_text is None:
            link_text = url
        link_text = link_text.replace("_", "\_")
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_overview=Widget.Markdown(
                title="Overview",
                is_header=True,
                formats=[
                    info.get("deploy_params", {}).get("checkpoint_name"),
                    info.get("architecture"),
                    info.get("task_type"),
                    info.get("runtime"),
                    info.get("hardware"),
                    url,
                    link_text,
                    self._loader.docs_link,
                    self._loader.gt_project_info.id,
                    self._loader.gt_project_info.name
                ],
            ),
            markdown_key_metrics=Widget.Markdown(
                title="Key Metrics",
                is_header=True,
                formats=[
                    self._loader.vis_texts.definitions.average_precision,
                    self._loader.vis_texts.definitions.confidence_threshold,
                    self._loader.vis_texts.definitions.confidence_score,
                ],
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go  # pylint: disable=import-error

        # Overall Metrics
        base_metrics = self._loader.mp.base_metrics()
        r = list(base_metrics.values())
        theta = [self._loader.mp.metric_names[k] for k in base_metrics.keys()]
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=r + [r[0]],
                theta=theta + [theta[0]],
                fill="toself",
                name="Overall Metrics",
                hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[0.0, 1.0],
                    ticks="outside",
                ),
                angularaxis=dict(rotation=90, direction="clockwise"),
            ),
            dragmode=False,
            # title="Overall Metrics",
            # width=700,
            # height=500,
            # autosize=False,
            margin=dict(l=25, r=25, t=25, b=25),
        )
        fig.update_layout(
            modebar=dict(
                remove=[
                    "zoom2d",
                    "pan2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                ]
            )
        )
        return fig
