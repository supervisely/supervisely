from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class RenormalizedErrorOverUnion(BaseVisMetrics):

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            "renormalized_error_over_union",
            "Renormalized Error over Union",
            text=self.vis_texts.markdown_renormalized_error_ou,
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget("intersection_error_over_union", self.get_figure())

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        labels = ["Boundary EoU", "Extent EoU", "Segment EoU"]

        for idx, eval_result in enumerate(self.eval_results, 1):
            model_name = f"[{idx}] {eval_result.short_name}"

            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=[
                        eval_result.mp.boundary_renormed_eou,
                        eval_result.mp.extent_renormed_eou,
                        eval_result.mp.segment_renormed_eou,
                    ],
                    name=model_name,
                    text=[
                        eval_result.mp.boundary_renormed_eou,
                        eval_result.mp.extent_renormed_eou,
                        eval_result.mp.segment_renormed_eou,
                    ],
                    textposition="outside",
                    marker=dict(color=eval_result.color, line=dict(width=0.7)),
                    width=0.4,
                )
            )

        fig.update_traces(hovertemplate="%{x}: %{y:.2f}<extra></extra>")
        fig.update_layout(
            barmode="group",
            bargap=0.15,
            bargroupgap=0.05,
            width=700 if len(self.eval_results) < 4 else 1000,
        )

        return fig
