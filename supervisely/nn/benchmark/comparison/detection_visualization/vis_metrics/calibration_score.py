from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    CollapseWidget,
    MarkdownWidget,
    NotificationWidget,
    TableWidget,
)


class CalibrationScore(BaseVisMetrics):
    @property
    def header_md(self) -> MarkdownWidget:
        text_template = self.vis_texts.markdown_calibration_score_1
        text = text_template.format(self.vis_texts.definitions.confidence_score)
        return MarkdownWidget(
            name="markdown_calibration_score",
            title="Calibration Score",
            text=text,
        )

    @property
    def collapse_tip(self) -> CollapseWidget:
        md = MarkdownWidget(
            name="what_is_calibration",
            title="What is calibration?",
            text=self.vis_texts.markdown_what_is_calibration,
        )
        return CollapseWidget([md])

    @property
    def header_md_2(self) -> MarkdownWidget:
        return MarkdownWidget(
            name="markdown_calibration_score_2",
            title="",
            text=self.vis_texts.markdown_calibration_score_2,
        )

    @property
    def table(self) -> TableWidget:
        columns = [" ", "confidence threshold", "ECE", "MCE"]
        columns_options = [
            {"disableSort": True},
            {"disableSort": True},
            {"disableSort": True},
            {"disableSort": True},
        ]
        content = []
        for i, eval_result in enumerate(self.eval_results):
            name = f"[{i+1}] {eval_result.name}"
            conf_threshold = eval_result.mp.m_full.get_f1_optimal_conf()[0] or 0.0
            ece = eval_result.mp.m_full.expected_calibration_error()
            mce = eval_result.mp.m_full.maximum_calibration_error()
            row = [name, round(conf_threshold, 2), round(ece, 2), round(mce, 2)]
            dct = {
                "row": row,
                "id": name,
                "items": row,
            }
            content.append(dct)
        data = {
            "columns": columns,
            "columnsOptions": columns_options,
            "content": content,
        }
        return TableWidget(
            "table_reliability",
            data,
            show_header_controls=False,
            # main_column=columns[0],
            fix_columns=1,
        )

    @property
    def reliability_diagram_md(self) -> MarkdownWidget:
        return MarkdownWidget(
            name="markdown_reliability_diagram",
            title="Reliability Diagram",
            text=self.vis_texts.markdown_reliability_diagram,
        )

    @property
    def reliability_chart(self) -> ChartWidget:
        return ChartWidget(name="chart_reliability", figure=self.get_rel_figure())

    @property
    def collapse_ece(self) -> CollapseWidget:
        md = MarkdownWidget(
            name="markdown_calibration_curve_interpretation",
            title="How to interpret the Calibration curve",
            text=self.vis_texts.markdown_calibration_curve_interpretation,
        )
        return CollapseWidget([md])

    @property
    def confidence_score_md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_confidence_score_1.format(
            self.vis_texts.definitions.confidence_threshold
        )
        return MarkdownWidget(
            "markdown_confidence_score_1",
            "Confidence Score Profile",
            text,
        )

    @property
    def confidence_chart(self) -> ChartWidget:
        return ChartWidget(name="chart_confidence", figure=self.get_conf_figure())

    @property
    def confidence_score_md_2(self) -> MarkdownWidget:
        return MarkdownWidget(
            name="markdown_confidence_score_2",
            title="",
            text=self.vis_texts.markdown_confidence_score_2,
        )

    @property
    def collapse_conf_score(self) -> CollapseWidget:
        md = MarkdownWidget(
            name="markdown_plot_confidence_profile",
            title="How to plot Confidence Profile?",
            text=self.vis_texts.markdown_plot_confidence_profile,
        )
        return CollapseWidget([md])

    def get_conf_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        # Create an empty figure
        fig = go.Figure()

        for i, eval_result in enumerate(self.eval_results):
            # Add a line trace for each eval_result
            fig.add_trace(
                go.Scatter(
                    x=eval_result.dfsp_down["scores"],
                    y=eval_result.dfsp_down["f1"],
                    mode="lines",
                    name=f"[{i+1}] {eval_result.name}",
                    line=dict(color=eval_result.color),
                    hovertemplate="Confidence Score: %{x:.2f}<br>Value: %{y:.2f}<extra></extra>",
                )
            )

            # Add a vertical line and annotation for F1-optimal threshold if available
            if eval_result.mp.f1_optimal_conf is not None and eval_result.mp.best_f1 is not None:
                fig.add_shape(
                    type="line",
                    x0=eval_result.mp.f1_optimal_conf,
                    x1=eval_result.mp.f1_optimal_conf,
                    y0=0,
                    y1=eval_result.mp.best_f1,
                    line=dict(color="gray", width=2, dash="dash"),
                    name=f"F1-optimal threshold ({eval_result.name})",
                )
            if eval_result.mp.custom_conf_threshold is not None:
                fig.add_shape(
                    type="line",
                    x0=eval_result.mp.custom_conf_threshold,
                    x1=eval_result.mp.custom_conf_threshold,
                    y0=0,
                    y1=eval_result.mp.custom_f1,
                    line=dict(color="black", width=2, dash="dash"),
                    name=f"Confidence threshold ({eval_result.name})",
                )

        # Update the layout
        fig.update_layout(
            yaxis=dict(range=[0, 1], title="Scores"),
            xaxis=dict(range=[0, 1], tick0=0, dtick=0.1, title="Confidence Score"),
            height=500,
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
            showlegend=True,  # Show legend to differentiate between results
        )

        return fig

    def get_rel_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        for i, eval_result in enumerate(self.eval_results):
            # Calibration curve (only positive predictions)
            true_probs, pred_probs = eval_result.mp.m_full.calibration_curve()

            fig.add_trace(
                go.Scatter(
                    x=pred_probs,
                    y=true_probs,
                    mode="lines+markers",
                    name=f"[{i+1}] {eval_result.name}",
                    line=dict(color=eval_result.color),
                    hovertemplate=f"{eval_result.name}<br>"
                    + "Confidence Score: %{x:.2f}<br>Fraction of True Positives: %{y:.2f}<extra></extra>",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfectly calibrated",
                line=dict(color="orange", dash="dash"),
            )
        )

        fig.update_layout(
            # title="Calibration Curve (only positive predictions)",
            xaxis_title="Confidence Score",
            yaxis_title="Fraction of True Positives",
            legend=dict(x=0.6, y=0.1),
            xaxis=dict(range=[0, 1.1]),
            yaxis=dict(range=[0, 1.1]),
            width=700,
            height=500,
        )
        return fig
