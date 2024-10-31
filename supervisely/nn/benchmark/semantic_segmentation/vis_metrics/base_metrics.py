from supervisely.nn.benchmark.comparison.detection_visualization.vis_metrics.vis_metric import (
    BaseVisMetric,
)
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget


class BaseMetrics(BaseVisMetric):
    def __init__(self, vis_texts, eval_result: SemanticSegmentationEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget("base_metrics_chart", self.get_figure())

    def get_figure(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Basic segmentation metrics",
                "Intersection & Error over Union",
                "Renormalized Error over Union",
            ),
            specs=[[{"type": "polar"}, {"type": "domain"}, {"type": "xy"}]],
        )
        # first subplot
        categories = [
            "mPixel accuracy",
            "mPrecision",
            "mRecall",
            "mF1-score",
            "mIoU",
            "mBoundaryIoU",
            "mPixel accuracy",
        ]
        values = [
            self.eval_result.mp.pixel_accuracy,
            self.eval_result.mp.precision,
            self.eval_result.mp.recall,
            self.eval_result.mp.f1_score,
            self.eval_result.mp.iou,
            self.eval_result.mp.boundary_iou,
            self.eval_result.mp.pixel_accuracy,
        ]
        trace_1 = go.Scatterpolar(
            mode="lines+text",
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="cornflowerblue",
            line_color="blue",
            opacity=0.6,
            text=values,
            textposition=[
                "bottom right",
                "top center",
                "top center",
                "middle left",
                "bottom center",
                "bottom right",
                "bottom right",
            ],
            textfont=dict(color="blue"),
        )
        fig.add_trace(trace_1, row=1, col=1)
        # second subplot
        labels = ["mIoU", "mBoundaryEoU", "mExtentEoU", "mSegmentEoU"]
        values = [
            self.eval_result.mp.iou,
            self.eval_result.mp.boundary_eou,
            self.eval_result.mp.extent_eou,
            self.eval_result.mp.segment_eou,
        ]
        trace_2 = go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            textposition="outside",
            textinfo="percent+label",
            marker=dict(colors=["cornflowerblue", "moccasin", "lightgreen", "orangered"]),
        )
        fig.add_trace(trace_2, row=1, col=2)
        # third subplot
        labels = ["boundary", "extent", "segment"]
        values = [
            self.eval_result.mp.boundary_renormed_eou,
            self.eval_result.mp.extent_renormed_eou,
            self.eval_result.mp.segment_renormed_eou,
        ]
        trace_3 = go.Bar(
            x=labels,
            y=values,
            orientation="v",
            text=values,
            width=[0.5, 0.5, 0.5],
            textposition="outside",
            marker_color=["moccasin", "lightgreen", "orangered"],
        )
        fig.add_trace(trace_3, row=1, col=3)
        fig.update_layout(
            height=400,
            width=1200,
            polar=dict(
                radialaxis=dict(visible=True, showline=False, showticklabels=False, range=[0, 100])
            ),
            showlegend=False,
            plot_bgcolor="rgba(0, 0, 0, 0)",
            yaxis=dict(showticklabels=False),
            yaxis_range=[0, int(max(values)) + 4],
        )
        fig.layout.annotations[0].update(y=1.2)
        fig.layout.annotations[1].update(y=1.2)
        fig.layout.annotations[2].update(y=1.2)

        return fig
