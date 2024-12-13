from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class IntersectionErrorOverUnion(BaseVisMetrics):

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            "intersection_error_over_union",
            "Intersection & Error Over Union",
            text=self.vis_texts.markdown_iou,
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget("intersection_error_over_union", self.get_figure())

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error
        from plotly.subplots import make_subplots  # pylint: disable=import-error

        labels = ["mIoU", "mBoundaryEoU", "mExtentEoU", "mSegmentEoU"]

        length = len(self.eval_results)
        cols = 3 if length > 2 else 2
        cols = 4 if length % 4 == 0 else cols
        rows = length // cols + (1 if length % cols != 0 else 0)

        fig = make_subplots(rows=rows, cols=cols, specs=[[{"type": "domain"}] * cols] * rows)

        annotations = []
        for idx, eval_result in enumerate(self.eval_results, start=1):
            col = idx % cols + (cols if idx % cols == 0 else 0)
            row = idx // cols + (1 if idx % cols != 0 else 0)

            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=[
                        eval_result.mp.iou,
                        eval_result.mp.boundary_eou,
                        eval_result.mp.extent_eou,
                        eval_result.mp.segment_eou,
                    ],
                    hole=0.5,
                    textposition="outside",
                    textinfo="percent+label",
                    marker=dict(colors=["#8ACAA1", "#FFE4B5", "#F7ADAA", "#dd3f3f"]),
                ),
                row=row,
                col=col,
            )
            
            text = f"[{idx}] {eval_result.name[:7]}"
            text += "..." if len(eval_result.name) > 7 else ""
            annotations.append(
                dict(
                    text=text,
                    x=sum(fig.get_subplot(row, col).x) / 2,
                    y=sum(fig.get_subplot(row, col).y) / 2,
                    showarrow=False,
                    xanchor="center",
                )
            )
        fig.update_layout(annotations=annotations)

        return fig
