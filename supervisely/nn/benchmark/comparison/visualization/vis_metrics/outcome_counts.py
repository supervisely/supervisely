from supervisely.nn.benchmark.comparison.visualization.vis_metrics.vis_metric import (
    BaseVisMetric,
)
from supervisely.nn.benchmark.comparison.visualization.widgets import ChartWidget


class OutcomeCounts(BaseVisMetric):

    CHART_MAIN = "chart_outcome_counts"
    CHART_COMPARISON = "chart_outcome_counts_comparison"

    @property
    def chart_widget_main(self) -> ChartWidget:
        return ChartWidget(name=self.CHART_MAIN, figure=self.get_main_figure())
        # TODO: add click_data

    @property
    def chart_widget_comparison(self) -> ChartWidget:
        return ChartWidget(name=self.CHART_COMPARISON, figure=self.get_comparison_figure())
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
        tp_counts = [eval_result.mp.TP_count for eval_result in self.eval_results]
        fn_counts = [eval_result.mp.FN_count for eval_result in self.eval_results]
        fp_counts = [eval_result.mp.FP_count for eval_result in self.eval_results]
        model_names = [f"Model {idx}" for idx in range(1, len(self.eval_results) + 1)]
        counts = [tp_counts, fn_counts, fp_counts]
        names = ["TP", "FN", "FP"]
        colors = ["#8ACAA1", "#dd3f3f", "#F7ADAA"]

        for metric, values, color in zip(names, counts, colors):
            fig.add_trace(
                go.Bar(
                    x=values,
                    y=model_names,
                    name=metric,
                    orientation="h",
                    marker=dict(color=color),
                    hovertemplate=f"{metric}: %{{x}} objects<extra></extra>",
                )
            )

        fig = self.update_figure_layout(fig)
        return fig

    def get_comparison_figure(self):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go

        fig = go.Figure()

        common_tp, common_fp, common_fn, diff_tp, diff_fp, diff_fn = self.get_common_and_diffs()
        colors = ["#8ACAA1", "#dd3f3f", "#F7ADAA"]

        for idx in range(1, len(self.eval_results) + 1):
            y = f"Model {idx}"
            for metric, values, color in zip(
                ["TP", "FN", "FP"], [diff_tp, diff_fn, diff_fp], colors
            ):
                fig.add_trace(
                    go.Bar(
                        x=[len(values.get(f"Model {idx}", []))],
                        y=[y],
                        name=metric,
                        orientation="h",
                        marker=dict(color=color),
                        hovertemplate=f"{metric}: %{{x}} objects<extra></extra>",
                    )
                )

        for metric, values, color in zip(
            ["TP", "FN", "FP"], [common_tp, common_fn, common_fp], colors
        ):
            fig.add_trace(
                go.Bar(
                    x=[len(values)],
                    y=["Common"],
                    name=metric,
                    orientation="h",
                    marker=dict(color=color),
                    hovertemplate=f"{metric}: %{{x}} objects<extra></extra>",
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
                curr_model_tp[key] = m
            for m in eval_result.mp.m.fp_matches:
                img_id = m["image_id"]
                category_id = m["category_id"]
                key = f"{img_id}_{category_id}"
                curr_model_fp[key] = m
            for m in eval_result.mp.m.fn_matches:
                img_id = m["image_id"]
                category_id = m["category_id"]
                key = f"{img_id}_{category_id}"
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
