# pylint: disable=no-member
# pylint: disable=not-an-iterable

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Iterator, List, NamedTuple, Optional, Tuple

if TYPE_CHECKING:
    from supervisely.nn.benchmark.metric_loader import MetricLoader

from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template
from plotly.subplots import make_subplots

import supervisely.nn.benchmark.metric_texts as contents
from supervisely._utils import camel_to_snake, rand_str
from supervisely.api.image_api import ImageInfo
from supervisely.collection.str_enum import StrEnum
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_texts import definitions


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return self.func(owner)


class CVTask(StrEnum):

    OBJECT_DETECTION: str = "object_detection"
    SEGMENTATION: str = "segmentation"


template_markdown_str = """
            <sly-iw-markdown
              id="{{ widget_id }}"
              iw-widget-id="{{ widget_id }}"
              :actions="{
                'init': {
                  'dataSource': '{{ data_source }}',
                },
              }"
              :command="{{ command }}"
              :data="{{ data }}"
            />
"""

template_chart_str = """
            <sly-iw-chart
              iw-widget-id="{{ widget_id }}"{% if switchable %}
              v-show="state.{{ radio_group }} === '{{ switch_key }}'"
              {% endif %}:actions="{
                'init': {
                  'dataSource': '{{ init_data_source }}',
                },{% if chart_click_data_source %}
                'chart-click': {
                  'dataSource': '{{ chart_click_data_source }}',{% if cls_name in ['outcome_counts'] %}
                  'getKey': (payload) => payload.points[0].data.name,{% endif %}{% if cls_name in ['frequently_confused'] %}
                  'getKey': (payload) => payload.points[0].label,{% endif %}{% if cls_name in ['confusion_matrix'] %}
                  'keySeparator': '{{ key_separator }}',{% endif %}
                },{% endif %}
              }"
              :command="{{ command }}"
              :data="{{ data }}"
            />
"""

template_radiogroup_str = """<el-radio v-model="state.{{ radio_group }}" label="{{ switch_key }}">{{ switch_key }}</el-radio>"""


class BaseBidget:
    def __init__(self) -> None:
        self.type = camel_to_snake(self.__class__.__name__)
        self.id = f"{self.type}_{rand_str(5)}"
        self.name = None


class Widget:

    class Markdown(BaseBidget):

        def __init__(self, header: Optional[str] = None) -> None:
            self.is_before_chart = None  # see self.template_str
            self.header = header
            super().__init__()

    class Chart(BaseBidget):

        def __init__(self, switch_key: Optional[str] = None) -> None:
            self.switch_key = switch_key
            super().__init__()

    class Table(BaseBidget):

        def __init__(self) -> None:
            super().__init__()


class Schema:

    def __init__(self, **kwargs) -> None:
        for argname, widget in kwargs.items():
            widget.name = argname
            setattr(self, argname, widget)

    def __iter__(self) -> Iterator:
        for attr in vars(self).values():
            yield attr

    def __getitem__(self, key) -> Widget:
        return getattr(self, key)

    def __repr__(self):
        elements = ", ".join(f"{attr.name} ({attr.type})" for attr in self)
        return f"Schema({elements})"


class MetricVis:

    cv_tasks: Tuple[CVTask] = tuple(CVTask.values())
    clickable: bool = False
    switchable: bool = False

    _template_markdown = Template(template_markdown_str)
    _template_chart = Template(template_chart_str)
    _template_radiogroup = Template(template_radiogroup_str)
    _keypair_sep = "&-|-&"

    schema: Schema = None

    # pylint: disable=no-self-argument
    @classproperty
    def radiogroup_id(cls) -> Optional[str]:
        if cls.switchable:
            return f"radiogroup_" + cls.name

    # pylint: disable=no-self-argument
    @classproperty
    def template_sidebar_str(cls) -> str:
        res = ""
        for item in cls.schema:
            if isinstance(item, Widget.Markdown):
                if item.header is not None:
                    res += f"""\n          <div>\n            <el-button type="text" @click="data.scrollIntoView='{item.id}'">{item.header}</el-button>\n          </div>"""
        return res

    # pylint: disable=no-self-argument
    @classproperty
    def template_main_str(cls) -> str:
        res = ""
        _is_before_chart = True
        if cls.switchable:
            for widget in cls.schema:
                if isinstance(widget, Widget.Chart):
                    basename = f"{widget.name}_{cls.name}"
                    res += "\n            {{ " + f"el_radio_{basename}_html" + " }}"

        for widget in cls.schema:
            if isinstance(widget, Widget.Chart):
                _is_before_chart = False
            if isinstance(widget, Widget.Markdown):
                widget.is_before_chart = _is_before_chart

            if isinstance(widget, Widget.Markdown) and widget.is_before_chart:
                res += "\n            {{ " + f"{widget.name}_html" + " }}"
                continue

            if isinstance(widget, Widget.Chart):
                basename = f"{widget.name}_{cls.name}"
                res += "\n            {{ " + f"{basename}_html" + " }}"
                if cls.clickable:
                    res += "\n            {{ " + f"{basename}_clickdata_html" + " }}"
                continue

            if isinstance(widget, Widget.Markdown) and not widget.is_before_chart:
                res += "\n            {{ " + f"{widget.name}_html" + " }}"
                continue

        return res

    @classmethod
    def get_html_snippets(cls, loader: MetricLoader) -> dict:
        res = {}
        for widget in cls.schema:
            if isinstance(widget, Widget.Markdown):
                res[f"{widget.name}_html"] = cls._template_markdown.render(
                    {
                        "widget_id": widget.id,
                        "data_source": f"/data/{widget.name}.md",
                        "command": "command",
                        "data": "data",
                    }
                )

            if isinstance(widget, Widget.Chart):
                basename = f"{widget.name}_{cls.name}"
                if cls.switchable:
                    res[f"el_radio_{basename}_html"] = cls._template_radiogroup.render(
                        {
                            "radio_group": cls.radiogroup_id,
                            "switch_key": widget.switch_key,
                        }
                    )
                chart_click_path = f"/data/{basename}_clickdata.json" if cls.clickable else None
                res[f"{basename}_html"] = cls._template_chart.render(
                    {
                        "widget_id": widget.id,
                        "init_data_source": f"/data/{basename}.json",
                        "chart_click_data_source": chart_click_path,
                        "command": "command",
                        "data": "data",
                        "cls_name": cls.name,
                        "key_separator": cls._keypair_sep,
                        "switchable": cls.switchable,
                        "radio_group": cls.radiogroup_id,
                        "switch_key": widget.switch_key,
                    }
                )

        return res

    # pylint: disable=no-self-argument
    @classproperty
    def name(cls) -> str:
        # pylint: disable=no-member
        return camel_to_snake(cls.__name__)

    @classmethod
    def get_figure(cls, loader: MetricLoader, widget: Widget) -> Optional[go.Figure]:
        pass

    @classmethod
    def get_click_data(cls, loader: MetricLoader, widget: Widget.Chart) -> Optional[dict]:
        pass

    @classmethod
    def get_table(cls, loader: MetricLoader) -> Optional[dict]:
        pass

    @classmethod
    def _get_md_content(cls, widget: Widget):
        return getattr(contents, widget.name)

    @classmethod
    def get_md_content(cls, loader: MetricLoader, widget: Widget):
        # redefinable method
        return cls._get_md_content(widget)


class Overview(MetricVis):

    schema = Schema(
        markdown_overview=Widget.Markdown(header="Overview"),
        markdown_key_metrics=Widget.Markdown(header="Key Metrics"),
        chart=Widget.Chart(),
    )

    @classmethod
    def get_figure(cls, loader: MetricLoader, widget: Widget.Chart) -> Optional[go.Figure]:
        # Overall Metrics
        base_metrics = loader.m.base_metrics()
        r = list(base_metrics.values())
        theta = [metric_provider.METRIC_NAMES[k] for k in base_metrics.keys()]
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
                radialaxis=dict(range=[0.0, 1.0]),
                angularaxis=dict(rotation=90, direction="clockwise"),
            ),
            # title="Overall Metrics",
            width=600,
            height=500,
        )
        return fig

    @classmethod
    def get_md_content(cls, loader: MetricLoader, widget: Widget):
        res = cls._get_md_content(widget)
        if widget.name == cls.schema.markdown_key_metrics.name:  # pylint: disable=E1101
            return res.format(
                definitions.average_precision,
                definitions.confidence_threshold,
                definitions.confidence_score,
            )
        return res


class OutcomeCounts(MetricVis):

    clickable: bool = True

    schema = Schema(
        markdown_outcome_counts=Widget.Markdown(header="Outcome Counts"),
        chart=Widget.Chart(),
    )

    @classmethod
    def get_figure(cls, loader: MetricLoader, widget: Widget.Chart) -> Optional[go.Figure]:
        # Outcome counts
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[loader.m.TP_count],
                y=["Outcome"],
                name="TP",
                orientation="h",
                marker=dict(color="#1fb466"),
            )
        )
        fig.add_trace(
            go.Bar(
                x=[loader.m.FN_count],
                y=["Outcome"],
                name="FN",
                orientation="h",
                marker=dict(color="#dd3f3f"),
            )
        )
        fig.add_trace(
            go.Bar(
                x=[loader.m.FP_count],
                y=["Outcome"],
                name="FP",
                orientation="h",
                marker=dict(color="#d5a5a5"),
            )
        )
        fig.update_layout(
            barmode="stack",
            width=600,
            height=300,
        )
        fig.update_xaxes(title_text="Count")
        fig.update_yaxes(tickangle=-90)

        return fig

    @classmethod
    def get_click_data(cls, loader: MetricLoader, widget: Widget.Chart) -> Optional[dict]:
        res = {}
        res["projectMeta"] = loader.dt_project_meta.to_json()
        res["clickData"] = {}
        for key, v in loader.click_data.outcome_counts.items():
            res["clickData"][key] = {}
            res["clickData"][key]["layoutData"] = {}
            res["clickData"][key]["layout"] = []

            tmp = {0: [], 1: [], 2: [], 3: []}

            images = set(x["dt_img_id"] for x in v)

            for idx, img_id in enumerate(images):
                ui_id = f"ann_{img_id}"
                info: ImageInfo = loader.dt_images[img_id]
                res["clickData"][key]["layoutData"][ui_id] = {
                    "imageUrl": info.preview_url,
                    "annotation": {
                        "imageId": info.id,
                        "imageName": info.name,
                        "createdAt": info.created_at,
                        "updatedAt": info.updated_at,
                        "link": info.link,
                        "annotation": loader.dt_ann_jsons[img_id],
                    },
                }
                if len(tmp[3]) < 5:
                    tmp[idx % 4].append(ui_id)

            for _, val in tmp.items():
                res["clickData"][key]["layout"].append(val)

        return res

    @classmethod
    def get_md_content(cls, loader: MetricLoader, widget: Widget):
        res = cls._get_md_content(widget)
        return res.format(
            definitions.true_positives,
            definitions.false_positives,
            definitions.false_negatives,
        )


class Recall(MetricVis):
    schema = Schema(
        markdown_R=Widget.Markdown(header="Recall"),
        markdown_R_perclass=Widget.Markdown(),
        chart=Widget.Chart(),
    )

    @classmethod
    def get_figure(cls, loader: MetricLoader, widget: Widget.Chart) -> Optional[go.Figure]:
        # Per-class Precision bar chart
        # per_class_metrics_df_sorted = per_class_metrics_df.sort_values(by="recall")
        fig = px.bar(
            loader.per_class_metrics_sorted,
            x="category",
            y="recall",
            title="Per-class Recall (Sorted by F1)",
            color="recall",
            color_continuous_scale="Plasma",
        )
        if len(loader.per_class_metrics_sorted) <= 20:
            fig.update_traces(
                text=loader.per_class_metrics_sorted["recall"].round(2), textposition="outside"
            )
        fig.update_xaxes(title_text="Category")
        fig.update_yaxes(title_text="Recall", range=[0, 1])
        return fig

    @classmethod
    def get_md_content(cls, loader: MetricLoader, widget: Widget):
        res = cls._get_md_content(widget)
        if widget.name == cls.schema.markdown_R_perclass.name:
            return res.format(
                definitions.f1_score,
            )
        return res


class Precision(MetricVis):
    schema = Schema(
        markdown_P=Widget.Markdown(header="Precision"),
        markdown_P_perclass=Widget.Markdown(),
        chart=Widget.Chart(),
    )

    @classmethod
    def get_figure(cls, loader: MetricLoader, widget: Widget) -> Optional[go.Figure]:
        # Per-class Precision bar chart
        # per_class_metrics_df_sorted = per_class_metrics_df.sort_values(by="precision")
        fig = px.bar(
            loader.per_class_metrics_sorted,
            x="category",
            y="precision",
            title="Per-class Precision (Sorted by F1)",
            color="precision",
            color_continuous_scale="Plasma",
        )
        if len(loader.per_class_metrics_sorted) <= 20:
            fig.update_traces(
                text=loader.per_class_metrics_sorted["precision"].round(2),
                textposition="outside",
            )
        fig.update_xaxes(title_text="Category")
        fig.update_yaxes(title_text="Precision", range=[0, 1])
        return fig

    @classmethod
    def get_md_content(cls, loader: MetricLoader, widget: Widget):
        res = cls._get_md_content(widget)
        if widget.name == cls.schema.markdown_P_perclass.name:
            return res.format(
                definitions.f1_score,
            )
        return res


class RecallVsPrecision(MetricVis):
    schema = Schema(
        markdown_PR=Widget.Markdown(header="Recall vs Precision"),
        chart=Widget.Chart(),
    )

    @classmethod
    def get_figure(cls, loader: MetricLoader, widget: Widget.Chart) -> Optional[go.Figure]:
        blue_color = "#1f77b4"
        orange_color = "#ff7f0e"
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=loader.per_class_metrics_sorted["precision"],
                x=loader.per_class_metrics_sorted["category"],
                name="Precision",
                marker=dict(color=blue_color),
            )
        )
        fig.add_trace(
            go.Bar(
                y=loader.per_class_metrics_sorted["recall"],
                x=loader.per_class_metrics_sorted["category"],
                name="Recall",
                marker=dict(color=orange_color),
            )
        )
        fig.update_layout(
            barmode="group",
            title="Per-class Precision and Recall (Sorted by F1)",
        )
        fig.update_xaxes(title_text="Category")
        fig.update_yaxes(title_text="Value", range=[0, 1])
        # fig.show()
        return fig

    @classmethod
    def get_md_content(cls, loader: MetricLoader, widget: Widget):
        res = cls._get_md_content(widget)
        if widget.name == cls.schema.markdown_PR.name:
            return res.format(
                definitions.f1_score,
            )
        return res


class PRCurve(MetricVis):
    schema = Schema(
        markdown_pr_curve=Widget.Markdown(header="Precision-Recall Curve"),
        chart=Widget.Chart(),
    )

    @classmethod
    def get_figure(cls, loader: MetricLoader, widget: Widget.Chart) -> Optional[go.Figure]:
        # Precision-Recall curve
        fig = px.line(
            x=loader.m.recThrs,
            y=loader.m.pr_curve().mean(-1),
            # title="Precision-Recall Curve",
            labels={"x": "Recall", "y": "Precision"},
            width=600,
            height=500,
        )
        fig.data[0].name = "Model"
        fig.data[0].showlegend = True
        fig.update_traces(fill="tozeroy", line=dict(color="#1f77b4"))
        fig.add_trace(
            go.Scatter(
                x=loader.m.recThrs,
                y=[1] * len(loader.m.recThrs),
                name="Perfect",
                line=dict(color="orange", dash="dash"),
                showlegend=True,
            )
        )
        fig.add_annotation(
            text=f"mAP = {loader.m.base_metrics()['mAP']:.2f}",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.92,
            showarrow=False,
            bgcolor="white",
        )

        # fig.show()
        return fig

    @classmethod
    def get_md_content(cls, loader: MetricLoader, widget: Widget):
        res = cls._get_md_content(widget)
        if widget.name == cls.schema.markdown_pr_curve.name:
            return res.format(
                definitions.f1_score,
            )
        return res


class PRCurveByClass(MetricVis):
    schema = Schema(
        markdown_pr_by_class=Widget.Markdown(header="PR Curve by Class"),
        chart=Widget.Chart(),
    )

    @classmethod
    def get_figure(cls, loader: MetricLoader, widget: Widget.Chart) -> Optional[go.Figure]:

        # Precision-Recall curve per-class
        df = pd.DataFrame(loader.m.pr_curve(), columns=loader.m.cat_names)

        fig = px.line(
            df,
            x=loader.m.recThrs,
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


class ConfusionMatrix(MetricVis):

    clickable = True
    _keypair_sep = "-"

    schema = Schema(
        markdown_confusion_matrix=Widget.Markdown(header="Confusion Matrix"),
        chart=Widget.Chart(),
    )

    @classmethod
    def get_figure(cls, loader: MetricLoader, widget: Widget.Chart) -> Optional[go.Figure]:
        confusion_matrix = loader.m.confusion_matrix()
        # Confusion Matrix
        # TODO: Green-red
        cat_names = loader.m.cat_names
        none_name = "(None)"

        with np.errstate(divide="ignore"):
            loged_cm = np.log(confusion_matrix)

        df = pd.DataFrame(
            loged_cm,
            index=cat_names + [none_name],
            columns=cat_names + [none_name],
        )
        fig = px.imshow(
            df,
            labels=dict(x="Ground Truth", y="Predicted", color="Count"),
            # title="Confusion Matrix (log-scale)",
            width=1000,
            height=1000,
        )

        # Hover text
        fig.update_traces(
            customdata=confusion_matrix,
            hovertemplate="Count: %{customdata}<br>Predicted: %{y}<br>Ground Truth: %{x}",
        )

        # Text on cells
        if len(cat_names) <= 20:
            fig.update_traces(text=confusion_matrix, texttemplate="%{text}")

        # fig.show()
        return fig

    @classmethod
    def get_click_data(cls, loader: MetricLoader, widget: Widget.Chart) -> Optional[dict]:
        res = dict(projectMeta=loader.dt_project_meta.to_json())
        res["clickData"] = {}

        unique_pairs = set()
        filtered_pairs = []
        for k, v in loader.click_data.confusion_matrix.items():
            ordered_pair = tuple(sorted(k))
            if ordered_pair not in unique_pairs:
                unique_pairs.add(ordered_pair)
            else:
                continue

            subkey1, subkey2 = ordered_pair
            key = subkey1 + cls._keypair_sep + subkey2
            res["clickData"][key] = {}
            res["clickData"][key]["layoutData"] = {}
            res["clickData"][key]["layout"] = []

            tmp = {0: [], 1: [], 2: [], 3: []}
            images = set(x["dt_img_id"] for x in v)

            for idx, img_id in enumerate(images):
                ui_id = f"ann_{img_id}"
                info: ImageInfo = loader.dt_images[img_id]
                res["clickData"][key]["layoutData"][ui_id] = {
                    "imageUrl": info.preview_url,
                    "annotation": {
                        "imageId": info.id,
                        "imageName": info.name,
                        "createdAt": info.created_at,
                        "updatedAt": info.updated_at,
                        "link": info.link,
                        "annotation": loader.dt_ann_jsons[img_id],
                    },
                }
                if len(tmp[3]) < 5:
                    tmp[idx % 4].append(ui_id)

            for _, val in tmp.items():
                res["clickData"][key]["layout"].append(val)

        return res


class FrequentlyConfused(MetricVis):

    clickable: bool = True
    switchable: bool = True
    _keypair_sep: str = " - "

    schema = Schema(
        markdown_frequently_confused=Widget.Markdown(header="Frequently Confused Classes"),
        chart_01=Widget.Chart(switch_key="probability"),
        chart_02=Widget.Chart(switch_key="count"),
    )

    @classmethod
    def get_figure(cls, loader: MetricLoader, widget: Widget.Chart) -> Optional[Tuple[go.Figure]]:

        confusion_matrix = loader.m.confusion_matrix()

        # Frequency of confusion as bar chart
        confused_df = loader.m.frequently_confused(confusion_matrix, topk_pairs=20)
        confused_name_pairs = confused_df["category_pair"]
        x_labels = [f"{pair[0]} - {pair[1]}" for pair in confused_name_pairs]
        y_labels = confused_df[widget.switch_key]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=x_labels, y=y_labels, marker=dict(color=y_labels, colorscale="Reds"))
        )
        fig.update_layout(
            # title="Frequently confused class pairs",
            xaxis_title="Class pair",
            yaxis_title=y_labels.name.capitalize(),
        )
        fig.update_traces(text=y_labels.round(2))
        return fig

    @classmethod
    def get_md_content(cls, loader: MetricLoader, widget: Widget):
        res = cls._get_md_content(widget)
        if widget.name == cls.schema.markdown_frequently_confused.name:
            df = loader.m.frequently_confused(loader.m.confusion_matrix(), topk_pairs=20)
            pair = df["category_pair"][0]
            prob = df["probability"][0]
            return res.format(
                pair[0],
                pair[1],
                prob.round(2),
                pair[0],
                pair[1],
                (prob * 100).round(),
                pair[0],
                pair[1],
                pair[1],
                pair[0],
            )
        return res

    @classmethod
    def get_click_data(cls, loader: MetricLoader, widget: Widget.Chart) -> Optional[dict]:
        if not cls.clickable:
            return
        res = dict(projectMeta=loader.dt_project_meta.to_json())
        res["clickData"] = {}

        for keypair, v in loader.click_data.frequently_confused.items():

            subkey1, subkey2 = keypair
            key = subkey1 + cls._keypair_sep + subkey2
            res["clickData"][key] = {}
            res["clickData"][key]["layoutData"] = {}
            res["clickData"][key]["layout"] = []

            tmp = {0: [], 1: [], 2: [], 3: []}
            images = set(x["dt_img_id"] for x in v)

            for idx, img_id in enumerate(images):
                ui_id = f"ann_{img_id}"
                info: ImageInfo = loader.dt_images[img_id]
                res["clickData"][key]["layoutData"][ui_id] = {
                    "imageUrl": info.preview_url,
                    "annotation": {
                        "imageId": info.id,
                        "imageName": info.name,
                        "createdAt": info.created_at,
                        "updatedAt": info.updated_at,
                        "link": info.link,
                        "annotation": loader.dt_ann_jsons[img_id],
                    },
                }
                if len(tmp[3]) < 5:
                    tmp[idx % 4].append(ui_id)

            for _, val in tmp.items():
                res["clickData"][key]["layout"].append(val)

        return res


class IOUDistribution(MetricVis):

    @classmethod
    def get_figure(cls, loader: MetricLoader) -> Optional[go.Figure]:

        fig = go.Figure()
        nbins = 40
        fig.add_trace(go.Histogram(x=loader.m.ious, nbinsx=nbins))
        fig.update_layout(
            # title="IoU Distribution",
            xaxis_title="IoU",
            yaxis_title="Count",
            width=600,
            height=500,
        )

        # Add annotation for mean IoU as vertical line
        mean_iou = loader.m.ious.mean()
        y1 = len(loader.m.ious) // nbins
        fig.add_shape(
            type="line",
            x0=mean_iou,
            x1=mean_iou,
            y0=0,
            y1=y1,
            line=dict(color="orange", width=2, dash="dash"),
        )
        fig.add_annotation(x=mean_iou, y=y1, text=f"Mean IoU: {mean_iou:.2f}", showarrow=False)
        # fig.show()
        return fig


class ReliabilityDiagram(MetricVis):

    @classmethod
    def get_figure(cls, loader: MetricLoader) -> Optional[go.Figure]:
        # Calibration curve (only positive predictions)
        true_probs, pred_probs = loader.m_full.calibration_metrics.calibration_curve()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pred_probs,
                y=true_probs,
                mode="lines+markers",
                name="Calibration plot (Model)",
                line=dict(color="blue"),
                marker=dict(color="blue"),
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
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=700,
            height=500,
        )

        # fig.show()
        return fig


class ConfidenceScore(MetricVis):

    @classmethod
    def get_figure(cls, loader: MetricLoader) -> Optional[go.Figure]:

        color_map = {
            "Precision": "#1f77b4",
            "Recall": "orange",
        }
        fig = px.line(
            loader.dfsp_down,
            x="scores",
            y=["Precision", "Recall", "F1"],
            # title="Confidence Score Profile",
            labels={"value": "Value", "variable": "Metric", "scores": "Confidence Score"},
            width=None,
            height=500,
            color_discrete_map=color_map,
        )
        fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1], tick0=0, dtick=0.1))

        # Add vertical line for the best threshold
        fig.add_shape(
            type="line",
            x0=loader.f1_optimal_conf,
            x1=loader.f1_optimal_conf,
            y0=0,
            y1=loader.best_f1,
            line=dict(color="gray", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=loader.f1_optimal_conf,
            y=loader.best_f1 + 0.04,
            text=f"F1-optimal threshold: {loader.f1_optimal_conf:.2f}",
            showarrow=False,
        )
        # fig.show()
        return fig


class ConfidenceDistribution(MetricVis):

    @classmethod
    def get_figure(cls, loader: MetricLoader) -> Optional[go.Figure]:

        f1_optimal_conf, best_f1 = loader.m_full.get_f1_optimal_conf()

        # Histogram of confidence scores (TP vs FP)
        scores_tp, scores_fp = loader.m_full.calibration_metrics.scores_tp_and_fp(iou_idx=0)

        tp_y, tp_x = np.histogram(scores_tp, bins=40, range=[0, 1])
        fp_y, fp_x = np.histogram(scores_fp, bins=40, range=[0, 1])
        dx = (tp_x[1] - tp_x[0]) / 2

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=scores_fp,
                name="FP",
                marker=dict(color="#dd3f3f"),
                opacity=0.5,
                xbins=dict(size=0.025, start=0.0, end=1.0),
            )
        )
        fig.add_trace(
            go.Histogram(
                x=scores_tp,
                name="TP",
                marker=dict(color="#1fb466"),
                opacity=0.5,
                xbins=dict(size=0.025, start=0.0, end=1.0),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=tp_x + dx,
                y=tp_y,
                mode="lines+markers",
                name="TP",
                line=dict(color="#1fb466", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fp_x + dx,
                y=fp_y,
                mode="lines+markers",
                name="FP",
                line=dict(color="#dd3f3f", width=2),
            )
        )

        # Best threshold
        fig.add_shape(
            type="line",
            x0=f1_optimal_conf,
            x1=f1_optimal_conf,
            y0=0,
            y1=tp_y.max() * 1.3,
            line=dict(color="orange", width=1, dash="dash"),
        )
        fig.add_annotation(
            x=f1_optimal_conf,
            y=tp_y.max() * 1.3,
            text=f"F1-optimal threshold: {f1_optimal_conf:.2f}",
            showarrow=False,
        )

        fig.update_layout(
            barmode="overlay",
            title="Histogram of Confidence Scores (TP vs FP)",
            width=800,
            height=500,
        )
        fig.update_xaxes(title_text="Confidence Score", range=[0, 1])
        fig.update_yaxes(title_text="Count", range=[0, tp_y.max() * 1.3])
        # fig.show()
        return fig


class F1ScoreAtDifferentIOU(MetricVis):

    @classmethod
    def get_figure(cls, loader: MetricLoader) -> Optional[go.Figure]:
        # score_profile = loader.m_full.confidence_score_profile()
        f1s = loader.m_full.score_profile_f1s

        # downsample
        f1s_down = f1s[:, :: f1s.shape[1] // 1000]
        iou_names = list(map(lambda x: str(round(x, 2)), loader.m.iouThrs.tolist()))
        df = pd.DataFrame(
            np.concatenate([loader.dfsp_down["scores"].values[:, None], f1s_down.T], 1),
            columns=["scores"] + iou_names,
        )

        fig = px.line(
            df,
            x="scores",
            y=iou_names,
            # title="F1-Score at different IoU Thresholds",
            labels={"value": "Value", "variable": "IoU threshold", "scores": "Confidence Score"},
            color_discrete_sequence=px.colors.sequential.Viridis,
            width=None,
            height=500,
        )
        fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1], tick0=0, dtick=0.1))

        # add annotations for maximum F1-Score for each IoU threshold
        for i, iou in enumerate(iou_names):
            argmax_f1 = f1s[i].argmax()
            max_f1 = f1s[i][argmax_f1]
            score = loader.score_profile["scores"][argmax_f1]
            fig.add_annotation(
                x=score,
                y=max_f1,
                text=f"Best score: {score:.2f}",
                showarrow=True,
                arrowhead=1,
                arrowcolor="black",
                ax=0,
                ay=-30,
            )

        # fig.show()
        return fig


class PerClassAvgPrecision(MetricVis):

    @classmethod
    def get_figure(cls, loader: MetricLoader) -> Optional[go.Figure]:

        # AP per-class
        ap_per_class = loader.m.coco_precision[:, :, :, 0, 2].mean(axis=(0, 1))
        # Per-class Average Precision (AP)
        fig = px.scatter_polar(
            r=ap_per_class,
            theta=loader.m.cat_names,
            title="Per-class Average Precision (AP)",
            labels=dict(r="Average Precision", theta="Category"),
            width=800,
            height=800,
            range_r=[0, 1],
        )
        # fill points
        fig.update_traces(fill="toself")
        # fig.show()
        return fig


class PerClassOutcomeCounts(MetricVis):

    clickable: bool = True

    @classmethod
    def get_switchable_figures(cls, loader: MetricLoader) -> Optional[Tuple[go.Figure]]:
        # Per-class Counts
        iou_thres = 0

        tp = loader.m.true_positives[:, iou_thres]
        fp = loader.m.false_positives[:, iou_thres]
        fn = loader.m.false_negatives[:, iou_thres]

        # normalize
        support = tp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            tp_rel = tp / support
            fp_rel = fp / support
            fn_rel = fn / support

            # sort by f1
            sort_scores = 2 * tp / (2 * tp + fp + fn)

        K = len(loader.m.cat_names)
        sort_indices = np.argsort(sort_scores)
        cat_names_sorted = [loader.m.cat_names[i] for i in sort_indices]
        tp_rel, fn_rel, fp_rel = tp_rel[sort_indices], fn_rel[sort_indices], fp_rel[sort_indices]

        # Stacked per-class counts
        data = {
            "count": np.concatenate([tp_rel, fn_rel, fp_rel]),
            "type": ["TP"] * K + ["FN"] * K + ["FP"] * K,
            "category": cat_names_sorted * 3,
        }

        df = pd.DataFrame(data)

        color_map = {"TP": "#1fb466", "FN": "#dd3f3f", "FP": "#d5a5a5"}
        fig = px.bar(
            df,
            x="category",
            y="count",
            color="type",
            # title="Per-class Outcome Counts",
            labels={"count": "Total Count", "category": "Category"},
            color_discrete_map=color_map,
        )

        # fig.show()

        # Stacked per-class counts
        data = {
            "count": np.concatenate([tp[sort_indices], fn[sort_indices], fp[sort_indices]]),
            "type": ["TP"] * K + ["FN"] * K + ["FP"] * K,
            "category": cat_names_sorted * 3,
        }

        df = pd.DataFrame(data)

        color_map = {"TP": "#1fb466", "FN": "#dd3f3f", "FP": "#d5a5a5"}
        fig_ = px.bar(
            df,
            x="category",
            y="count",
            color="type",
            # title="Per-class Outcome Counts",
            labels={"count": "Total Count", "category": "Category"},
            color_discrete_map=color_map,
        )

        return (fig, fig_)


class OverallErrorAnalysis(MetricVis):

    cv_tasks: Tuple[CVTask] = (CVTask.SEGMENTATION.value,)

    @classmethod
    def get_figure(cls, loader: MetricLoader) -> Optional[go.Figure]:
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
        values = [64, 60.4, 52, 51.4, 37.9, 20.5, 64]
        trace_1 = go.Scatterpolar(
            mode="lines+text",
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="cornflowerblue",
            line_color="blue",
            opacity=0.6,
            text=[64, 60.4, 52, 51.4, 37.9, 20.5, 64],
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
        values = [37.9, 13.1, 25.8, 23.2]
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
        values = [28.9, 37.6, 23.2]
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


class ClasswiseErrorAnalysis(MetricVis):

    cv_tasks: Tuple[CVTask] = (CVTask.SEGMENTATION.value,)

    @classmethod
    def get_figure(cls, loader: MetricLoader) -> Optional[go.Figure]:
        pd.options.mode.chained_assignment = None  # TODO rm later

        df = loader.result_df
        df.drop(["mean"], inplace=True)
        df = df[["IoU", "E_extent_oU", "E_boundary_oU", "E_segment_oU"]]
        df.sort_values(by="IoU", ascending=False, inplace=True)
        labels = list(df.index)
        color_palette = ["cornflowerblue", "moccasin", "lightgreen", "orangered"]

        fig = go.Figure()
        for i, column in enumerate(df.columns):
            fig.add_trace(
                go.Bar(
                    name=column,
                    y=df[column],
                    x=labels,
                    marker_color=color_palette[i],
                )
            )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(
            barmode="stack",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            title={
                "text": "Classwise segmentation error analysis",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
        )
        return fig
