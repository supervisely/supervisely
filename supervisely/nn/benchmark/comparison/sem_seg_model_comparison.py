import supervisely as sly
import string
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.figure_factory as ff
import numpy as np
import os


class SemanticSegmentationComparison:
    def __init__(self, eval_dirs, model_names=None):
        self.eval_dirs = eval_dirs
        self.n_models = len(self.eval_dirs)
        if not model_names:
            alphabet = list(string.ascii_uppercase)
            self.model_names = [f"model_{alphabet[i]}" for i in range(self.n_models)]
        else:
            self.model_names = model_names
        self.model_colors = ["cornflowerblue", "orangered", "springgreen", "yellow"]
        self.chart_width = 1200
        self.result_dfs = []
        for eval_dir in self.eval_dirs:
            result_df = pd.read_csv(f"{eval_dir}/result_df.csv", index_col="Unnamed: 0")
            self.result_dfs.append(result_df)

    def create_comparison_charts(self, output_dir="comparison_results"):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.create_basic_metrics_chart(output_dir)
        self.create_iou_eou_chart(output_dir)
        self.create_renormed_eou_chart(output_dir)
        self.create_classwise_error_chart(output_dir)
        self.create_freq_confused_chart(output_dir)
        basic_metrics_chart_img = sly.image.read(f"{output_dir}/basic_metrics.png")
        iou_eou_chart_img = sly.image.read(f"{output_dir}/iou_eou.png")
        eou_renormed_chart_img = sly.image.read(f"{output_dir}/eou_renormed.png")
        classwise_error_chart_img = sly.image.read(f"{output_dir}/classwise_error.png")
        freq_confused_chart_img = sly.image.read(f"{output_dir}/frequently_confused.png")
        sly.image.write(
            f"{output_dir}/total_dashboard.png",
            np.vstack(
                (
                    basic_metrics_chart_img,
                    iou_eou_chart_img,
                    eou_renormed_chart_img,
                    classwise_error_chart_img,
                    freq_confused_chart_img,
                )
            ),
        )

    def create_basic_metrics_chart(self, output_dir):
        basic_metric_categories = [
            "mPixel accuracy",
            "mPrecision",
            "mRecall",
            "mF1-score",
            "mIoU",
            "mBoundary IoU",
        ]
        fig = go.Figure()
        for i, df in enumerate(self.result_dfs):
            num_classes = len(df.index) - 1
            precision = round(df.loc["mean", "precision"] * 100, 1)
            recall = round(df.loc["mean", "recall"] * 100, 1)
            f1_score = round(df.loc["mean", "F1_score"] * 100, 1)
            iou = round(df.loc["mean", "IoU"] * 100, 1)
            boundary_iou = round(df.loc["mean", "boundary_IoU"] * 100, 1)
            overall_TP = df["TP"][:num_classes].sum()
            overall_FN = df["FN"][:num_classes].sum()
            pixel_accuracy = round((overall_TP / (overall_TP + overall_FN)) * 100, 1)
            basic_metric_values = [
                pixel_accuracy,
                precision,
                recall,
                f1_score,
                iou,
                boundary_iou,
            ]
            fig.add_trace(
                go.Bar(
                    name=self.model_names[i],
                    y=basic_metric_values,
                    x=basic_metric_categories,
                    marker_color=self.model_colors[i],
                    text=basic_metric_values,
                )
            )
        fig.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            title={
                "text": "Basic segmentation metrics comparison",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            yaxis=dict(showticklabels=False),
            yaxis_range=[0, 110],
            legend=dict(x=1.0, y=0.7),
            height=400,
            width=self.chart_width,
        )
        fig.update_traces(
            textposition="outside",
        )
        fig.write_image(f"{output_dir}/basic_metrics.png", scale=2.0)
        fig.write_html(f"{output_dir}/basic_metrics.html")

    def create_iou_eou_chart(self, output_dir):
        iou_eou_categories = ["mIoU", "mBoundaryEoU", "mExtentEoU", "mSegmentEoU"]
        fig = go.Figure()

        for i, df in enumerate(self.result_dfs):
            iou = round(df.loc["mean", "IoU"] * 100, 1)
            boundary_eou = round(df.loc["mean", "E_boundary_oU"] * 100, 1)
            extent_eou = round(df.loc["mean", "E_extent_oU"] * 100, 1)
            segment_eou = round(df.loc["mean", "E_segment_oU"] * 100, 1)
            iou_eou_values = [iou, boundary_eou, extent_eou, segment_eou]
            fig.add_trace(
                go.Bar(
                    name=self.model_names[i],
                    y=iou_eou_values,
                    x=iou_eou_categories,
                    marker_color=self.model_colors[i],
                    text=iou_eou_values,
                )
            )
        fig.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            title={
                "text": "Intersection & Error over Union comparison",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            yaxis=dict(showticklabels=False),
            yaxis_range=[0, 110],
            legend=dict(x=1.0, y=0.7),
            height=400,
            width=self.chart_width,
        )
        fig.update_traces(
            textposition="outside",
        )
        fig.write_image(f"{output_dir}/iou_eou.png", scale=2.0)
        fig.write_html(f"{output_dir}/iou_eou.html")

    def create_renormed_eou_chart(self, output_dir):
        renormed_eou_categories = ["boundary", "extent", "segment"]
        fig = go.Figure()

        for i, df in enumerate(self.result_dfs):
            boundary_renormed_eou = round(df.loc["mean", "E_boundary_oU_renormed"] * 100, 1)
            extent_renormed_eou = round(df.loc["mean", "E_extent_oU_renormed"] * 100, 1)
            segment_renormed_eou = round(df.loc["mean", "E_segment_oU_renormed"] * 100, 1)
            renormed_eou_values = [
                boundary_renormed_eou,
                extent_renormed_eou,
                segment_renormed_eou,
            ]
            fig.add_trace(
                go.Bar(
                    name=self.model_names[i],
                    y=renormed_eou_values,
                    x=renormed_eou_categories,
                    marker_color=self.model_colors[i],
                    text=renormed_eou_values,
                )
            )
        fig.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            title={
                "text": "Renormalized Error over Union comparison",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            yaxis=dict(showticklabels=False),
            yaxis_range=[0, 110],
            legend=dict(x=1.0, y=0.7),
            height=400,
            width=self.chart_width,
        )
        fig.update_traces(
            textposition="outside",
        )
        fig.write_image(f"{output_dir}/eou_renormed.png", scale=2.0)
        fig.write_html(f"{output_dir}/eou_renormed.html")

    def create_classwise_error_chart(self, output_dir):
        if len(self.result_dfs) == 2:
            n_rows = 1
            n_cols = 2
            specs = [[{"type": "xy"}, {"type": "xy"}]]
            chart_height = 400
            legend_y = -0.5
            title_y = 0.9
        if len(self.result_dfs) == 3:
            n_rows = len(self.result_dfs)
            n_cols = 1
            specs = [[{"type": "xy"}] for row in range(n_rows)]
            chart_height = 1000
            legend_y = -0.1
            title_y = 0.97
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=self.model_names,
            specs=specs,
        )

        for i, df in enumerate(self.result_dfs):
            df = df.drop(["mean"])
            if len(df.index) > 7:
                per_class_iou = df["IoU"].copy()
                per_class_iou.sort_values(ascending=True, inplace=True)
                target_classes = per_class_iou.index[:7].tolist()
                title_text = "Classwise segmentation error analysis<br><sup>(7 classes with highest error rates)</sup>"
                labels = target_classes[::-1]
                bar_data = df.loc[target_classes].copy()
            else:
                title_text = "Classwise segmentation error analysis"
                bar_data = df.copy()
            bar_data = bar_data[["IoU", "E_extent_oU", "E_boundary_oU", "E_segment_oU"]]
            bar_data.sort_values(by="IoU", ascending=False, inplace=True)
            if not len(df.index) > 7:
                labels = list(bar_data.index)
            color_palette = ["cornflowerblue", "moccasin", "lightgreen", "orangered"]
            showlegend = True if i == 0 else False

            for j, column in enumerate(bar_data.columns):
                if len(self.result_dfs) == 2:
                    fig.add_trace(
                        go.Bar(
                            name=column,
                            y=bar_data[column],
                            x=labels,
                            marker_color=color_palette[j],
                            showlegend=showlegend,
                        ),
                        row=1,
                        col=i + 1,
                    )
                else:
                    fig.add_trace(
                        go.Bar(
                            name=column,
                            y=bar_data[column],
                            x=labels,
                            marker_color=color_palette[j],
                            showlegend=showlegend,
                        ),
                        row=i + 1,
                        col=1,
                    )

        fig.update_layout(
            barmode="stack",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            title={
                "text": title_text,
                "y": title_y,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            height=chart_height,
            width=self.chart_width,
            legend=dict(yanchor="bottom", xanchor="center", orientation="h", y=legend_y, x=0.5),
        )
        fig.write_image(f"{output_dir}/classwise_error.png", scale=2.0)
        fig.write_html(f"{output_dir}/classwise_error.html")

    def create_freq_confused_chart(self, output_dir):
        if len(self.eval_dirs) == 2:
            chart_height = 700
            title_y = 0.95
        elif len(self.eval_dirs) == 3:
            chart_height = 1600
            title_y = 0.98
        confusion_matrixes = []
        for eval_dir in self.eval_dirs:
            confusion_matrix = np.load(f"{eval_dir}/confusion_matrix.npy")
            confusion_matrixes.append(confusion_matrix)
        class_names = list(self.result_dfs[0].index)[:-1]
        n_pairs = 10
        n_rows = len(self.eval_dirs)
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            subplot_titles=self.model_names,
            specs=[[{"type": "xy"}] for row in range(n_rows)],
        )

        for i, cm in enumerate(confusion_matrixes):
            non_diagonal_indexes = {}
            for j, idx in enumerate(np.ndindex(cm.shape)):
                if idx[0] != idx[1]:
                    non_diagonal_indexes[j] = idx

            indexes_1d = np.argsort(cm, axis=None)
            indexes_2d = [
                non_diagonal_indexes[idx] for idx in indexes_1d if idx in non_diagonal_indexes
            ][-n_pairs:]
            indexes_2d = np.asarray(indexes_2d[::-1])

            rows = indexes_2d[:, 0]
            cols = indexes_2d[:, 1]
            probs = cm[rows, cols]

            confused_classes = []
            for idx in indexes_2d:
                gt_idx, pred_idx = idx[0], idx[1]
                gt_class = class_names[gt_idx]
                pred_class = class_names[pred_idx]
                confused_classes.append(f"{gt_class}-{pred_class}")

            fig.add_trace(
                go.Bar(
                    x=confused_classes,
                    y=probs,
                    orientation="v",
                    text=probs,
                ),
                row=i + 1,
                col=1,
            )

        fig.update_traces(
            textposition="outside",
            marker=dict(color=probs, colorscale="orrd"),
        )
        fig.update_layout(
            title={
                "text": "Frequently confused classes comparison",
                "y": title_y,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            showlegend=False,
            height=chart_height,
            width=self.chart_width,
            plot_bgcolor="rgba(0, 0, 0, 0)",
        )
        for row in range(1, len(self.eval_dirs) + 1):
            fig.update_yaxes(showticklabels=False, row=row, col=1, range=[0, 1.1])
        fig.write_image(f"{output_dir}/frequently_confused.png", scale=2.0)
        fig.write_html(f"{output_dir}/frequently_confused.html")
