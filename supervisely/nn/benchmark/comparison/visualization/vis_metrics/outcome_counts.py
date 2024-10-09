from typing import List

from supervisely.nn.benchmark.comparison.evaluation_result import EvalResult
from supervisely.nn.benchmark.comparison.visualization.vis_metrics.vis_metric import (
    BaseVisMetric,
)
from supervisely.nn.benchmark.comparison.visualization.widgets import ChartWidget


class OutcomeCounts(BaseVisMetric):

    CHART_MAIN = "chart_outcome_counts"
    CHART_COMPARISON = "chart_outcome_counts_comparison"

    def __init__(self, vis_texts, eval_results: List[EvalResult]) -> None:
        """
        Class to create widgets for the outcome counts section.

        chart_widget_main property returns ChartWidget with Bar charts for each model with TP, FP, FN counts.
        chart_widget_comparison property returns ChartWidget with Bar charts with common and different TP, FP, FN counts.
        """
        super().__init__(vis_texts, eval_results)

        self.figure_main = self.get_main_figure()
        self.figure_comparison = self.get_comparison_figure()

    @property
    def chart_widget_main(self) -> ChartWidget:
        return ChartWidget(name=self.CHART_MAIN, figure=self.figure_main)
        # TODO: add click_data

    @property
    def chart_widget_comparison(self) -> ChartWidget:
        return ChartWidget(name=self.CHART_COMPARISON, figure=self.figure_comparison)
        # TODO: add click_data

    def update_figure_layout(self, fig):
        fig.update_layout(
            barmode="stack",
            width=600,
            height=300,
        )
        fig.update_xaxes(title_text="Count (objects)")
        fig.update_yaxes(tickangle=-90)

        fig.update_layout(
            dragmode=False,
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
            ),
        )
        return fig

    def get_main_figure(self):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        for idx, eval_result in enumerate(self.eval_results, 1):
            y = f"Model {idx}"
            fig.add_trace(
                go.Bar(
                    x=[eval_result.mp.TP_count],
                    y=[y],
                    name="TP",
                    orientation="h",
                    marker=dict(color="#8ACAA1"),
                    hovertemplate="TP: %{x} objects<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=[eval_result.mp.FN_count],
                    y=[y],
                    name="FN",
                    orientation="h",
                    marker=dict(color="#dd3f3f"),
                    hovertemplate="FN: %{x} objects<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=[eval_result.mp.FP_count],
                    y=[y],
                    name="FP",
                    orientation="h",
                    marker=dict(color="#F7ADAA"),
                    hovertemplate="FP: %{x} objects<extra></extra>",
                )
            )
        fig = self.update_figure_layout(fig)
        return fig

    def get_comparison_figure(self):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go

        fig = go.Figure()

        common_tp, common_fp, common_fn, diff_tp, diff_fp, diff_fn = self.get_common_and_diffs()
        for idx, eval_result in enumerate(self.eval_results, 1):
            y = f"Model {idx}"
            fig.add_trace(
                go.Bar(
                    x=[len(diff_tp.get(f"Model {idx}", []))],
                    y=[y],
                    name="TP",
                    orientation="h",
                    marker=dict(color="#8ACAA1"),
                    hovertemplate="TP: %{x} objects<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=[len(diff_fn.get(f"Model {idx}", []))],
                    y=[y],
                    name="FN",
                    orientation="h",
                    marker=dict(color="#dd3f3f"),
                    hovertemplate="FN: %{x} objects<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=[len(diff_fp.get(f"Model {idx}", []))],
                    y=[y],
                    name="FP",
                    orientation="h",
                    marker=dict(color="#F7ADAA"),
                    hovertemplate="FP: %{x} objects<extra></extra>",
                )
            )

        fig.add_trace(
            go.Bar(
                x=[len(common_tp)],
                y=["Common"],
                name="TP",
                orientation="h",
                marker=dict(color="#8ACAA1"),
                hovertemplate="TP: %{x} objects<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=[len(common_fn)],
                y=["Common"],
                name="FN",
                orientation="h",
                marker=dict(color="#dd3f3f"),
                hovertemplate="FN: %{x} objects<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=[len(common_fp)],
                y=["Common"],
                name="FP",
                orientation="h",
                marker=dict(color="#F7ADAA"),
                hovertemplate="FP: %{x} objects<extra></extra>",
            )
        )

        fig = self.update_figure_layout(fig)
        return fig

    def get_common_and_diffs(self):
        all_models_tp = {}
        all_models_fp = {}
        all_models_fn = {}
        for idx, eval_result in enumerate(self.eval_results, 1):
            curr_model_tp = all_models_tp.setdefault(f"Model {idx}", {})
            curr_model_fp = all_models_fp.setdefault(f"Model {idx}", {})
            curr_model_fn = all_models_fn.setdefault(f"Model {idx}", {})
            for m in eval_result.mp.m.tp_matches:
                img_id = m["image_id"]
                category_id = m["category_id"]
                key = f"{img_id}_{category_id}"
                if m["type"] == "TP":
                    curr_model_tp[key] = m
                elif m["type"] == "FP":
                    curr_model_fp[key] = m
                elif m["type"] == "FN":
                    curr_model_fn[key] = m

        common_tp = []  # list of tuples (match1, match2, ...)
        common_fp = []  # list of tuples (match1, match2, ...)
        common_fn = []  # list of tuples (match1, match2, ...)
        diff_tp = {}  # {model1: [match1, match2, ...], model2: [match1, match2, ...], ...}
        diff_fp = {}  # {model1: [match1, match2, ...], model2: [match1, match2, ...], ...}
        diff_fn = {}  # {model1: [match1, match2, ...], model2: [match1, match2, ...], ...}

        for model, tp_matches in all_models_tp.items():
            for key, match in tp_matches.items():
                if all(key in others_tp for others_tp in all_models_tp.values()):
                    common_tp.append([match] + [matches[key] for matches in all_models_tp.values()])
                else:
                    diff_tp.setdefault(model, []).append(match)

        for model, fp_matches in all_models_fp.items():
            for key, match in fp_matches.items():
                if all(key in others_fp for others_fp in all_models_fp.values()):
                    common_fp.append([match] + [matches[key] for matches in all_models_fp.values()])
                else:
                    diff_fp.setdefault(model, []).append(match)

        for model, fn_matches in all_models_fn.items():
            for key, match in fn_matches.items():
                if all(key in others_fn for others_fn in all_models_fn.values()):
                    common_fn.append([match] + [matches[key] for matches in all_models_fn.values()])
                else:
                    diff_fn.setdefault(model, []).append(match)

        return common_tp, common_fp, common_fn, diff_tp, diff_fp, diff_fn
