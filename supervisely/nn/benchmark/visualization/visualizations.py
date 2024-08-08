# pylint: disable=no-member
# pylint: disable=not-an-iterable

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, List, NamedTuple, Optional, Tuple, Union

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template
from plotly.subplots import make_subplots

import supervisely.nn.benchmark.visualization.vis_texts as contents
from supervisely._utils import camel_to_snake, rand_str
from supervisely.api.image_api import ImageInfo
from supervisely.collection.str_enum import StrEnum
from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.nn.benchmark.visualization.vis_texts import definitions
from supervisely.project.project_meta import ProjectMeta


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return self.func(owner)


# class CVTask(StrEnum):

#     OBJECT_DETECTION: str = "object_detection"
#     SEGMENTATION: str = "segmentation"


template_markdown_str = """
            <sly-iw-markdown
              id="{{ widget_id }}"
              class="markdown-no-border"
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
                  'keySeparator': '{{ key_separator }}',{% endif %}{% if cls_name in ['per_class_outcome_counts'] %}
                  'getKey': (payload) => payload.points[0].data.x,{% endif %}
                },{% endif %}
              }"
              :command="{{ command }}"
              :data="{{ data }}"
            />
"""

template_radiogroup_str = """<el-radio v-model="state.{{ radio_group }}" label="{{ switch_key }}">{{ switch_key }}</el-radio>"""


template_gallery_str = """<sly-iw-gallery
              iw-widget-id="{{ widget_id }}"{% if is_table_gallery %}
              ref='{{ widget_id }}'
              {% endif %}:actions="{
                'init': {
                  'dataSource': '{{ init_data_source }}',
                },
              }"{% if gallery_click_data_source %}
                'chart-click': {
                  'dataSource': '{{ gallery_click_data_source }}',
                  'limit':10,
                },{% endif %}              
              :command="{{ command }}"
              :data="{{ data }}"
              {% if gallery_click_data_source %}:options="{'isModalWindow': true}"{% endif %}  
            />"""

template_table_str = """<sly-iw-table
                iw-widget-id="{{ widget_id }}"
                style="cursor: pointer;"
                :options="{ isRowClickable: true }" 
                :actions="{
                  'init': {
                    'dataSource': '{{ init_data_source }}',
                  },
                  'chart-click': {
                    'dataSource': '{{ table_click_data }}',
                    'galleryId': '{{ table_gallery_id }}',                    
                    'getKey':(payload)=>payload.row[0],
                   },
                }"
              :command="{{ command }}"
              :data="{{ data }}"
            />"""

template_notification_str = """
            <sly-iw-notification              
              iw-widget-id="{{ widget_id }}"
              :data="{{ data }}"
            >
              <span slot="title">
                {{ title }}
              </span>

              <span slot="description">
                {{ description }}
              </span>
            </sly-iw-notification>"""


class BaseWidget:
    def __init__(self) -> None:
        self.type = camel_to_snake(self.__class__.__name__)
        self.id = f"{self.type}_{rand_str(5)}"
        self.name = None


class Widget:

    class Collapse(BaseWidget):

        def __init__(self, schema: Schema) -> None:
            super().__init__()
            self.schema = schema
            res = "<el-collapse>"
            for subwidget in schema:
                res += f"""\n                <el-collapse-item title="{subwidget.title}">"""
                res += "\n            {{ " + f"{subwidget.name}_html" + " }}"
                res += "\n                </el-collapse-item>"
            res += "\n            </el-collapse>"
            self.template_schema = Template(res)

    class Markdown(BaseWidget):

        def __init__(
            self, title: Optional[str] = None, is_header: bool = False, formats: list = []
        ) -> None:
            self.title = title
            self.is_header = is_header
            self.formats = formats
            super().__init__()

    class Notification(BaseWidget):

        def __init__(self, formats_title: list = [], formats_desc: list = []) -> None:
            self.title: str = None
            self.description: str = None
            self.formats_title: list = formats_title
            self.formats_desc: list = formats_desc
            super().__init__()

    class Chart(BaseWidget):

        def __init__(self, switch_key: Optional[str] = None) -> None:
            self.switch_key = switch_key
            super().__init__()

    class Table(BaseWidget):

        def __init__(self) -> None:
            from supervisely.app.widgets.fast_table.fast_table import FastTable

            self.table = FastTable
            self.gallery_id = None
            super().__init__()

    class Gallery(BaseWidget):

        def __init__(self, is_table_gallery: bool = False) -> None:
            from supervisely.app.widgets.grid_gallery_v2.grid_gallery_v2 import (
                GridGalleryV2,
            )

            self.is_table_gallery = is_table_gallery

            self.gallery = GridGalleryV2(
                columns_number=3,
                enable_zoom=False,
                default_tag_filters=[{"confidence": [0.6, 1]}, {"outcome": "TP"}],
                show_zoom_slider=False,
            )

            super().__init__()


class Schema:

    def __init__(self, **kw_widgets: Widget) -> None:
        for argname, widget in kw_widgets.items():
            widget.name = argname
            if isinstance(widget, Widget.Notification):
                widget.title = getattr(contents, argname)["title"]
                widget.description = getattr(contents, argname)["description"]
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

    def __init__(self, loader: Visualizer) -> None:

        self.cv_tasks: List[CVTask] = CVTask.values()
        self.clickable: bool = False
        self.switchable: bool = False
        self.schema: Schema = None

        self._loader = loader
        self._template_markdown = Template(template_markdown_str)
        self._template_chart = Template(template_chart_str)
        self._template_radiogroup = Template(template_radiogroup_str)
        self._template_gallery = Template(template_gallery_str)
        self._template_table = Template(template_table_str)
        self._template_notification = Template(template_notification_str)
        self._keypair_sep = "-"

    @property
    def radiogroup_id(self) -> Optional[str]:
        if self.switchable:
            return f"radiogroup_" + self.name

    @property
    def template_sidebar_str(self) -> str:
        res = ""
        for widget in self.schema:
            if isinstance(widget, Widget.Markdown):
                if widget.title is not None and widget.is_header:
                    res += f"""\n          <div>\n            <el-button type="text" @click="data.scrollIntoView='{widget.id}'">{widget.title}</el-button>\n          </div>"""
        return res

    @property
    def template_main_str(self) -> str:
        res = ""
        _is_before_chart = True

        def _add_radio_buttons():
            res = ""
            for widget in self.schema:
                if isinstance(widget, Widget.Chart):
                    basename = f"{widget.name}_{self.name}"
                    res += "\n            {{ " + f"el_radio_{basename}_html" + " }}"
            return res

        is_radiobuttons_added = False

        for widget in self.schema:
            if isinstance(widget, Widget.Chart):
                _is_before_chart = False

            if (
                isinstance(widget, (Widget.Markdown, Widget.Notification, Widget.Collapse))
                and _is_before_chart
            ):
                res += "\n            {{ " + f"{widget.name}_html" + " }}"
                continue

            if isinstance(widget, (Widget.Chart, Widget.Gallery, Widget.Table)):
                basename = f"{widget.name}_{self.name}"
                if self.switchable and not is_radiobuttons_added:
                    res += _add_radio_buttons()
                    is_radiobuttons_added = True
                res += "\n            {{ " + f"{basename}_html" + " }}"
                if self.clickable:
                    res += "\n            {{ " + f"{basename}_clickdata_html" + " }}"
                continue

            if (
                isinstance(widget, (Widget.Markdown, Widget.Notification, Widget.Collapse))
                and not _is_before_chart
            ):
                res += "\n            {{ " + f"{widget.name}_html" + " }}"
                continue

        return res

    def get_html_snippets(self) -> dict:
        res = {}
        for widget in self.schema:
            if isinstance(widget, Widget.Markdown):
                res[f"{widget.name}_html"] = self._template_markdown.render(
                    {
                        "widget_id": widget.id,
                        "data_source": f"/data/{widget.name}.md",
                        "command": "command",
                        "data": "data",
                    }
                )

            if isinstance(widget, Widget.Collapse):
                subres = {}
                for subwidget in widget.schema:
                    if isinstance(subwidget, Widget.Markdown):
                        subres[f"{subwidget.name}_html"] = self._template_markdown.render(
                            {
                                "widget_id": subwidget.id,
                                "data_source": f"/data/{subwidget.name}.md",
                                "command": "command",
                                "data": "data",
                            }
                        )
                res[f"{widget.name}_html"] = widget.template_schema.render(**subres)
                continue

            if isinstance(widget, Widget.Notification):
                res[f"{widget.name}_html"] = self._template_notification.render(
                    {
                        "widget_id": widget.id,
                        "data": "data",
                        "title": widget.title.format(*widget.formats_title),
                        "description": widget.description.format(*widget.formats_desc),
                    }
                )

            if isinstance(widget, Widget.Chart):
                basename = f"{widget.name}_{self.name}"
                if self.switchable:
                    res[f"el_radio_{basename}_html"] = self._template_radiogroup.render(
                        {
                            "radio_group": self.radiogroup_id,
                            "switch_key": widget.switch_key,
                        }
                    )
                chart_click_path = f"/data/{basename}_clickdata.json" if self.clickable else None
                res[f"{basename}_html"] = self._template_chart.render(
                    {
                        "widget_id": widget.id,
                        "init_data_source": f"/data/{basename}.json",
                        "chart_click_data_source": chart_click_path,
                        "command": "command",
                        "data": "data",
                        "cls_name": self.name,
                        "key_separator": self._keypair_sep,
                        "switchable": self.switchable,
                        "radio_group": self.radiogroup_id,
                        "switch_key": widget.switch_key,
                    }
                )
            if isinstance(widget, Widget.Gallery):
                basename = f"{widget.name}_{self.name}"
                if widget.is_table_gallery:
                    for w in self.schema:
                        if isinstance(w, Widget.Table):
                            w.gallery_id = widget.id

                gallery_click_data_source = (
                    f"/data/{basename}_click_data.json" if self.clickable else None
                )
                res[f"{basename}_html"] = self._template_gallery.render(
                    {
                        "widget_id": widget.id,
                        "init_data_source": f"/data/{basename}.json",
                        "command": "command",
                        "data": "data",
                        "is_table_gallery": widget.is_table_gallery,
                        "gallery_click_data_source": gallery_click_data_source,
                    }
                )

            if isinstance(widget, Widget.Table):
                basename = f"{widget.name}_{self.name}"
                res[f"{basename}_html"] = self._template_table.render(
                    {
                        "widget_id": widget.id,
                        "init_data_source": f"/data/{basename}.json",
                        "command": "command",
                        "data": "data",
                        "table_click_data": f"/data/{widget.name}_{self.name}_click_data.json",
                        "table_gallery_id": widget.gallery_id,
                    }
                )

        return res

    @property
    def name(self) -> str:
        return camel_to_snake(self.__class__.__name__)

    def get_figure(self, widget: Widget.Chart) -> Optional[go.Figure]:
        pass

    def get_click_data(self, widget: Widget.Chart) -> Optional[dict]:
        pass

    def get_table(self, widget: Widget.Table) -> Optional[dict]:
        pass

    def get_gallery(self, widget: Widget.Gallery) -> Optional[dict]:
        pass

    def get_gallery_click_data(self, widget: Widget.Gallery) -> Optional[dict]:
        pass

    def get_md_content(self, widget: Widget.Markdown):
        return getattr(contents, widget.name).format(*widget.formats)

    def initialize_formats(self, loader: Visualizer, widget: Widget):
        pass


class Overview(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_overview=Widget.Markdown(title="Overview", is_header=True),
            markdown_key_metrics=Widget.Markdown(
                title="Key Metrics",
                is_header=True,
                formats=[
                    definitions.average_precision,
                    definitions.confidence_threshold,
                    definitions.confidence_score,
                ],
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart) -> Optional[go.Figure]:
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
            # margin=dict(l=50, r=50, t=100, b=50, pad=4),
        )
        # fig.update_polars(automargin=True)
        fig.update_yaxes(automargin="top")
        # fig.update_xaxes(automargin="top")
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


class ExplorerGrid(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.clickable = True
        self.schema = Schema(
            markdown_explorer=Widget.Markdown(title="Explore Predictions", is_header=True),
            gallery=Widget.Gallery(),
        )

    def get_gallery(self, widget: Widget.Gallery):
        res = {}
        api = self._loader._api
        gt_project_id = self._loader.gt_project_info.id
        dt_project_id = self._loader.dt_project_info.id
        diff_project_id = self._loader.diff_project_info.id
        gt_dataset = api.dataset.get_list(gt_project_id)[0]
        dt_dataset = api.dataset.get_list(dt_project_id)[0]
        diff_dataset = api.dataset.get_list(diff_project_id)[0]
        gt_image_infos = api.image.get_list(dataset_id=gt_dataset.id)[:5]
        pred_image_infos = api.image.get_list(dataset_id=dt_dataset.id)[:5]
        diff_image_infos = api.image.get_list(dataset_id=diff_dataset.id)[:5]
        project_metas = [
            ProjectMeta.from_json(data=api.project.get_meta(id=x))
            for x in [gt_project_id, dt_project_id, diff_project_id]
        ]
        for gt_image, pred_image, diff_image in zip(
            gt_image_infos, pred_image_infos, diff_image_infos
        ):
            image_infos = [gt_image, pred_image, diff_image]
            ann_infos = [api.annotation.download(x.id) for x in image_infos]

            for idx, (image_info, ann_info, project_meta) in enumerate(
                zip(image_infos, ann_infos, project_metas)
            ):
                image_name = image_info.name
                image_url = image_info.full_storage_url
                is_ignore = True if idx in [0, 1] else False
                widget.gallery.append(
                    title=image_name,
                    image_url=image_url,
                    annotation_info=ann_info,
                    column_index=idx,
                    project_meta=project_meta,
                    ignore_tags_filtering=is_ignore,
                )
        res.update(widget.gallery.get_json_state())
        res.update(widget.gallery.get_json_data()["content"])
        res["layoutData"] = res.pop("annotations")
        res["projectMeta"] = project_metas[1].to_json()

        return res

    def get_gallery_click_data(self, widget: Widget.Table) -> Optional[dict]:
        res = {}
        res["layoutTemplate"] = [
            {"skipObjectsFiltering": True},
            {"skipObjectsFiltering": True},
            None,
        ]
        res["clickData"] = {}

        l1 = list(self._loader.gt_images_dct.values())
        l2 = list(self._loader.dt_images_dct.values())
        l3 = list(self._loader.diff_images_dct.values())

        for gt, dt, diff in zip(l1, l2, l3):
            res["clickData"][gt.name] = dict(imagesIds=[gt.id, dt.id, diff.id])

        return res


class ModelPredictions(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_predictions_gallery=Widget.Markdown(title="Model Predictions", is_header=True),
            gallery=Widget.Gallery(is_table_gallery=True),
            markdown_predictions_table=Widget.Markdown(title="Prediction Table", is_header=True),
            table=Widget.Table(),
        )
        self._row_ids = None

    def get_gallery(self, widget: Widget.Gallery) -> dict:
        res = {}
        api = self._loader._api
        df = self._loader.mp.prediction_table().round(2)
        gt_project_id = self._loader.gt_project_info.id
        dt_project_id = self._loader.dt_project_info.id
        diff_project_id = self._loader.diff_project_info.id

        def _get_image(name: str, project_id):
            for dataset in api.dataset.get_list(project_id):
                info = api.image.get_info_by_name(dataset.id, name)
                if info is None:
                    continue
                else:
                    return info

        infos = [None] * 3
        selected_image_name = str(df.iloc[0, 0])
        for idx, project_id in enumerate([gt_project_id, dt_project_id, diff_project_id]):
            infos[idx] = _get_image(selected_image_name, project_id)

        gt_image_info, pred_image_info, diff_image_info = infos

        images_infos = [gt_image_info, pred_image_info, diff_image_info]
        anns_infos = [api.annotation.download(x.id) for x in images_infos]
        project_metas = [
            ProjectMeta.from_json(data=api.project.get_meta(id=x))
            for x in [gt_project_id, dt_project_id, diff_project_id]
        ]

        for idx, (image_info, ann_info, project_meta) in enumerate(
            zip(images_infos, anns_infos, project_metas)
        ):
            image_name = image_info.name
            image_url = image_info.full_storage_url
            is_ignore = True if idx in [0, 1] else False
            widget.gallery.append(
                title=image_name,
                image_url=image_url,
                annotation_info=ann_info,
                column_index=idx,
                project_meta=project_meta,
                ignore_tags_filtering=is_ignore,
            )
        res.update(widget.gallery.get_json_state())
        res.update(widget.gallery.get_json_data()["content"])
        res["layoutData"] = res.pop("annotations")

        res["projectMeta"] = project_metas[1].to_json()

        return res

    def get_table(self, widget: Widget.Table) -> dict:
        res = {}
        api = self._loader._api
        gt_project_id = self._loader.gt_project_info.id
        dt_project_id = self._loader.dt_project_info.id
        diff_project_id = self._loader.diff_project_info.id
        gt_dataset = api.dataset.get_list(gt_project_id)[0]
        dt_dataset = api.dataset.get_list(dt_project_id)[0]
        diff_dataset = api.dataset.get_list(diff_project_id)[0]

        tmp = api.image.get_list(dataset_id=dt_dataset.id)
        df = self._loader.mp.prediction_table().round(2)
        df = df[df["image_name"].isin([x.name for x in tmp])]
        columns_options = [{}] * len(df.columns)
        for idx, col in enumerate(columns_options):
            if idx == 0:
                continue
            columns_options[idx] = {"maxValue": df.iloc[:, idx].max()}
        table_model_preds = widget.table(df, columns_options=columns_options)
        tbl = table_model_preds.to_json()

        res["columns"] = tbl["columns"]
        res["columnsOptions"] = columns_options
        res["content"] = []

        key_mapping = {}
        for old, new in zip(ImageInfo._fields, api.image.info_sequence()):
            key_mapping[old] = new

        self._row_ids = []

        for row in tbl["data"]["data"]:
            name = row["items"][0]
            info = self._loader.dt_images_dct_by_name[name]

            dct = {
                "row": {key_mapping[k]: v for k, v in info._asdict().items()},
                "id": info.name,
                "items": row["items"],
            }

            self._row_ids.append(dct["id"])
            res["content"].append(dct)

        return res

    def get_table_click_data(self, widget: Widget.Table) -> Optional[dict]:
        res = {}
        res["layoutTemplate"] = [
            {"skipObjectsFiltering": True},
            {"skipObjectsFiltering": True},
            None,
        ]
        res["clickData"] = {}

        for idx_str in self._row_ids:

            gt_image = self._loader.gt_images_dct_by_name[idx_str]
            dt_image = self._loader.dt_images_dct_by_name[idx_str]
            diff_image = self._loader.diff_images_dct_by_name[idx_str]

            res["clickData"][idx_str] = dict(imagesIds=[gt_image.id, dt_image.id, diff_image.id])

        return res


class WhatIs(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_what_is=Widget.Markdown(title="What is YOLOv8 model", is_header=True),
            markdown_experts=Widget.Markdown(title="Expert Insights", is_header=True),
            markdown_how_to_use=Widget.Markdown(
                title="How To Use: Training, Inference, Evaluation Loop", is_header=True
            ),
        )


class OutcomeCounts(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)

        self.clickable: bool = True
        self.schema = Schema(
            markdown_outcome_counts=Widget.Markdown(
                title="Outcome Counts",
                is_header=True,
                formats=[
                    definitions.true_positives,
                    definitions.false_positives,
                    definitions.false_negatives,
                ],
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart) -> Optional[go.Figure]:
        # Outcome counts
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[self._loader.mp.TP_count],
                y=["Outcome"],
                name="TP",
                orientation="h",
                marker=dict(color="#1fb466"),
            )
        )
        fig.add_trace(
            go.Bar(
                x=[self._loader.mp.FN_count],
                y=["Outcome"],
                name="FN",
                orientation="h",
                marker=dict(color="#dd3f3f"),
            )
        )
        fig.add_trace(
            go.Bar(
                x=[self._loader.mp.FP_count],
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

    def get_click_data(self, widget: Widget.Chart) -> Optional[dict]:
        res = {}
        res["filters"] = [
            {"type": "tag", "tagId": "confidence", "value": [0.6, 1]},
            {"type": "tag", "tagId": "outcome", "value": "TP"},
        ]
        res["options"] = {
            "fitOnResize": True,
            "enableZoom": False,
            "showOpacityInHeader": True,
            "showZoomInHeader": True,
            "opacity": 0.8,
            "enableObjectsPointerEvents": True,
            "showTransparentBackground": True,
            "showFilter": True,
        }
        res["projectMeta"] = self._loader.dt_project_meta.to_json()
        res["clickData"] = {}
        for key, v in self._loader.click_data.outcome_counts.items():
            res["clickData"][key] = {}
            # res["clickData"][key]["layoutData"] = {}
            # res["clickData"][key]["layout"] = []

            res["clickData"][key]["imagesIds"] = []

            tmp = {0: [], 1: [], 2: [], 3: []}

            images = set(x["dt_img_id"] for x in v)

            for idx, img_id in enumerate(images):
                # ui_id = f"ann_{img_id}"
                # info: ImageInfo = self._loader.dt_images_dct[img_id]
                res["clickData"][key]["imagesIds"].append(img_id)
            #     res["clickData"][key]["layoutData"][ui_id] = {
            #         "imageUrl": info.preview_url,
            #         "annotation": {
            #             "imageId": info.id,
            #             "imageName": info.name,
            #             "createdAt": info.created_at,
            #             "updatedAt": info.updated_at,
            #             "link": info.link,
            #             "annotation": self._loader.dt_ann_jsons[img_id],
            #         },
            #     }
            #     if len(tmp[3]) < 5:
            #         tmp[idx % 4].append(ui_id)

            # for _, val in tmp.items():
            #     res["clickData"][key]["layout"].append(val)

        return res


class Recall(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        tp_plus_fn = self._loader.mp.TP_count + self._loader.mp.FN_count
        self.schema = Schema(
            markdown_R=Widget.Markdown(title="Recall", is_header=True),
            notification_recall=Widget.Notification(
                formats_title=[self._loader.base_metrics()["recall"].round(2)],
                formats_desc=[self._loader.mp.TP_count, tp_plus_fn],
            ),
            markdown_R_perclass=Widget.Markdown(formats=[definitions.f1_score]),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart) -> Optional[go.Figure]:
        # Per-class Precision bar chart
        # per_class_metrics_df_sorted = per_class_metrics_df.sort_values(by="recall")
        sorted_by_f1 = self._loader.mp.per_class_metrics().sort_values(by="f1")
        fig = px.bar(
            sorted_by_f1,
            x="category",
            y="recall",
            # title="Per-class Recall (Sorted by F1)",
            color="recall",
            color_continuous_scale="Plasma",
        )
        if len(sorted_by_f1) <= 20:
            fig.update_traces(
                text=sorted_by_f1["recall"].round(2),
                textposition="outside",
            )
        fig.update_xaxes(title_text="Category")
        fig.update_yaxes(title_text="Recall", range=[0, 1])
        return fig


class Precision(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_P=Widget.Markdown(title="Precision", is_header=True),
            notification_precision=Widget.Notification(
                formats_title=[self._loader.base_metrics()["precision"].round(2)],
                formats_desc=[
                    self._loader.mp.TP_count,
                    (self._loader.mp.TP_count + self._loader.mp.FP_count),
                ],
            ),
            markdown_P_perclass=Widget.Markdown(formats=[definitions.f1_score]),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget) -> Optional[go.Figure]:
        # Per-class Precision bar chart
        # per_class_metrics_df_sorted = per_class_metrics_df.sort_values(by="precision")
        sorted_by_precision = self._loader.mp.per_class_metrics().sort_values(by="precision")
        fig = px.bar(
            sorted_by_precision,
            x="category",
            y="precision",
            # title="Per-class Precision (Sorted by F1)",
            color="precision",
            color_continuous_scale="Plasma",
        )
        if len(sorted_by_precision) <= 20:
            fig.update_traces(
                text=sorted_by_precision.round(2),
                textposition="outside",
            )
        fig.update_xaxes(title_text="Category")
        fig.update_yaxes(title_text="Precision", range=[0, 1])
        return fig


class RecallVsPrecision(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_PR=Widget.Markdown(
                title="Recall vs Precision", is_header=True, formats=[definitions.f1_score]
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart) -> Optional[go.Figure]:
        blue_color = "#1f77b4"
        orange_color = "#ff7f0e"
        sorted_by_f1 = self._loader.mp.per_class_metrics().sort_values(by="f1")
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=sorted_by_f1["precision"],
                x=sorted_by_f1["category"],
                name="Precision",
                marker=dict(color=blue_color),
            )
        )
        fig.add_trace(
            go.Bar(
                y=sorted_by_f1["recall"],
                x=sorted_by_f1["category"],
                name="Recall",
                marker=dict(color=orange_color),
            )
        )
        fig.update_layout(
            barmode="group",
            # title="Per-class Precision and Recall (Sorted by F1)",
        )
        fig.update_xaxes(title_text="Category")
        fig.update_yaxes(title_text="Value", range=[0, 1])
        # fig.show()
        return fig


class PRCurve(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_pr_curve=Widget.Markdown(
                title="Precision-Recall Curve", is_header=True, formats=[definitions.f1_score]
            ),
            collapse_pr=Widget.Collapse(
                schema=Schema(
                    markdown_trade_offs=Widget.Markdown(
                        title="About Trade-offs between precision and recall"
                    ),
                    markdown_what_is_pr_curve=Widget.Markdown(
                        title="What is PR curve?",
                        formats=[
                            definitions.confidence_score,
                            definitions.true_positives,
                            definitions.false_positives,
                        ],
                    ),
                )
            ),
            notification_ap=Widget.Notification(
                formats_title=[loader.base_metrics()["mAP"].round(2)]
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart) -> Optional[go.Figure]:
        # Precision-Recall curve
        fig = px.line(
            x=self._loader.mp.recThrs,
            y=self._loader.mp.pr_curve().mean(-1),
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
                x=self._loader.mp.recThrs,
                y=[1] * len(self._loader.mp.recThrs),
                name="Perfect",
                line=dict(color="orange", dash="dash"),
                showlegend=True,
            )
        )
        fig.add_annotation(
            text=f"mAP = {self._loader.mp.base_metrics()['mAP']:.2f}",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.92,
            showarrow=False,
            bgcolor="white",
        )
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
        # fig.show()
        return fig


class PRCurveByClass(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_pr_by_class=Widget.Markdown(title="PR Curve by Class"),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart) -> Optional[go.Figure]:

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


class ConfusionMatrix(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)

        self.clickable = True
        self.schema = Schema(
            markdown_confusion_matrix=Widget.Markdown(title="Confusion Matrix", is_header=True),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart) -> Optional[go.Figure]:
        confusion_matrix = self._loader.mp.confusion_matrix()
        # Confusion Matrix
        # TODO: Green-red
        cat_names = self._loader.mp.cat_names
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

    def get_click_data(self, widget: Widget.Chart) -> Optional[dict]:
        res = dict(projectMeta=self._loader.dt_project_meta.to_json())
        res["clickData"] = {}

        unique_pairs = set()
        filtered_pairs = []
        for k, v in self._loader.click_data.confusion_matrix.items():
            ordered_pair = tuple(sorted(k))
            if ordered_pair not in unique_pairs:
                unique_pairs.add(ordered_pair)
            else:
                continue

            subkey1, subkey2 = ordered_pair
            key = subkey1 + self._keypair_sep + subkey2
            res["clickData"][key] = {}
            res["clickData"][key]["layoutData"] = {}
            res["clickData"][key]["layout"] = []

            tmp = {0: [], 1: [], 2: [], 3: []}
            images = set(x["dt_img_id"] for x in v)

            for idx, img_id in enumerate(images):
                ui_id = f"ann_{img_id}"
                info: ImageInfo = self._loader.dt_images_dct[img_id]
                res["clickData"][key]["layoutData"][ui_id] = {
                    "imageUrl": info.preview_url,
                    "annotation": {
                        "imageId": info.id,
                        "imageName": info.name,
                        "createdAt": info.created_at,
                        "updatedAt": info.updated_at,
                        "link": info.link,
                        "annotation": self._loader.dt_ann_jsons[img_id],
                    },
                }
                if len(tmp[3]) < 5:
                    tmp[idx % 4].append(ui_id)

            for _, val in tmp.items():
                res["clickData"][key]["layout"].append(val)

        return res


class FrequentlyConfused(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)

        self.clickable: bool = True
        self.switchable: bool = True
        self._keypair_sep: str = " - "
        df = self._loader.mp.frequently_confused()
        pair = df["category_pair"][0]
        prob = df["probability"][0]
        self.schema = Schema(
            markdown_frequently_confused=Widget.Markdown(
                title="Frequently Confused Classes",
                is_header=True,
                formats=[
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
                ],
            ),
            chart_01=Widget.Chart(switch_key="probability"),
            chart_02=Widget.Chart(switch_key="count"),
        )

    def get_figure(self, widget: Widget.Chart) -> Optional[Tuple[go.Figure]]:

        # Frequency of confusion as bar chart
        confused_df = self._loader.mp.frequently_confused()
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

    def get_click_data(self, widget: Widget.Chart) -> Optional[dict]:
        if not self.clickable:
            return
        res = dict(projectMeta=self._loader.dt_project_meta.to_json())
        res["clickData"] = {}

        for keypair, v in self._loader.click_data.frequently_confused.items():

            subkey1, subkey2 = keypair
            key = subkey1 + self._keypair_sep + subkey2
            res["clickData"][key] = {}
            res["clickData"][key]["layoutData"] = {}
            res["clickData"][key]["layout"] = []

            tmp = {0: [], 1: [], 2: [], 3: []}
            images = set(x["dt_img_id"] for x in v)

            for idx, img_id in enumerate(images):
                ui_id = f"ann_{img_id}"
                info: ImageInfo = self._loader.dt_images_dct[img_id]
                res["clickData"][key]["layoutData"][ui_id] = {
                    "imageUrl": info.preview_url,
                    "annotation": {
                        "imageId": info.id,
                        "imageName": info.name,
                        "createdAt": info.created_at,
                        "updatedAt": info.updated_at,
                        "link": info.link,
                        "annotation": self._loader.dt_ann_jsons[img_id],
                    },
                }
                if len(tmp[3]) < 5:
                    tmp[idx % 4].append(ui_id)

            for _, val in tmp.items():
                res["clickData"][key]["layout"].append(val)

        return res


class IOUDistribution(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_localization_accuracy=Widget.Markdown(
                title="Localization Accuracy (IoU)", is_header=True
            ),
            collapse_iou=Widget.Collapse(
                Schema(markdown_iou_calculation=Widget.Markdown(title="How IoU is calculated?"))
            ),
            markdown_iou_distribution=Widget.Markdown(
                title="IoU Distribution", is_header=True, formats=[definitions.iou_score]
            ),
            chart=Widget.Chart(),
            notification_avg_iou=Widget.Notification(
                formats_title=[self._loader.base_metrics()["iou"].round(2)]
            ),
        )

    def get_figure(self, widget: Widget) -> Optional[go.Figure]:

        fig = go.Figure()
        nbins = 40
        fig.add_trace(go.Histogram(x=self._loader.mp.ious, nbinsx=nbins))
        fig.update_layout(
            # title="IoU Distribution",
            xaxis_title="IoU",
            yaxis_title="Count",
            width=600,
            height=500,
        )

        # Add annotation for mean IoU as vertical line
        mean_iou = self._loader.mp.ious.mean()
        y1 = len(self._loader.mp.ious) // nbins
        fig.add_shape(
            type="line",
            x0=mean_iou,
            x1=mean_iou,
            y0=0,
            y1=y1,
            line=dict(color="orange", width=2, dash="dash"),
        )
        fig.add_annotation(x=mean_iou, y=y1, text=f"Mean IoU: {mean_iou:.2f}", showarrow=False)
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


class ReliabilityDiagram(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_calibration_score_1=Widget.Markdown(
                title="Calibration Score", is_header=True, formats=[definitions.confidence_score]
            ),
            collapse_what_is=Widget.Collapse(
                Schema(markdown_what_is_calibration=Widget.Markdown(title="What is calibration?"))
            ),
            markdown_calibration_score_2=Widget.Markdown(),
            markdown_reliability_diagram=Widget.Markdown(
                title="Reliability Diagram", is_header=True
            ),
            chart=Widget.Chart(),
            collapse_ece=Widget.Collapse(
                Schema(
                    markdown_calibration_curve_interpretation=Widget.Markdown(
                        title="How to interpret the Calibration curve"
                    )
                )
            ),
            notification_ece=Widget.Notification(
                formats_title=[self._loader.mp.m_full.expected_calibration_error().round(4)]
            ),
        )

    def get_figure(self, widget: Widget) -> Optional[go.Figure]:
        # Calibration curve (only positive predictions)
        true_probs, pred_probs = self._loader.mp.m_full.calibration_curve()

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

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_confidence_score_1=Widget.Markdown(
                title="Confidence Score Profile",
                is_header=True,
                formats=[definitions.confidence_threshold],
            ),
            chart=Widget.Chart(),
            markdown_confidence_score_2=Widget.Markdown(),
            collapse_conf_score=Widget.Collapse(
                Schema(
                    markdown_plot_confidence_profile=Widget.Markdown(
                        title="How to plot Confidence Profile?"
                    )
                )
            ),
            markdown_calibration_score_3=Widget.Markdown(),
        )

    def get_figure(self, widget: Widget) -> Optional[go.Figure]:

        color_map = {
            "Precision": "#1f77b4",
            "Recall": "orange",
        }

        fig = px.line(
            self._loader.dfsp_down,
            x="scores",
            y=["Precision", "Recall", "F1"],
            # title="Confidence Score Profile",
            labels={"value": "Value", "variable": "Metric", "scores": "Confidence Score"},
            width=None,
            height=500,
            color_discrete_map=color_map,
        )
        fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1], tick0=0, dtick=0.1))

        if self._loader.mp.f1_optimal_conf is not None and self._loader.mp.best_f1 is not None:
            # Add vertical line for the best threshold
            fig.add_shape(
                type="line",
                x0=self._loader.mp.f1_optimal_conf,
                x1=self._loader.mp.f1_optimal_conf,
                y0=0,
                y1=self._loader.mp.best_f1,
                line=dict(color="gray", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=self._loader.mp.f1_optimal_conf,
                y=self._loader.mp.best_f1 + 0.04,
                text=f"F1-optimal threshold: {self._loader.mp.f1_optimal_conf:.2f}",
                showarrow=False,
            )
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


class F1ScoreAtDifferentIOU(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_f1_at_ious=Widget.Markdown(
                title="Confidence Profile at Different IoU thresholds",
                is_header=True,
                formats=[definitions.iou_threshold],
            ),
            notification_f1=Widget.Notification(
                formats_title=[round((self._loader.mp.m_full.get_f1_optimal_conf()[0] or 0.0), 4)]
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget) -> Optional[go.Figure]:
        # score_profile = self._loader.m_full.confidence_score_profile()
        f1s = self._loader.mp.m_full.score_profile_f1s

        # downsample
        if len(self._loader.df_score_profile) > 5000:
            f1s_down = f1s[:, :: f1s.shape[1] // 1000]
        else:
            f1s_down = f1s

        iou_names = list(map(lambda x: str(round(x, 2)), self._loader.mp.iouThrs.tolist()))
        df = pd.DataFrame(
            np.concatenate([self._loader.dfsp_down["scores"].values[:, None], f1s_down.T], 1),
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
            score = self._loader.mp.m_full.score_profile["scores"][argmax_f1]
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


class ConfidenceDistribution(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_confidence_distribution=Widget.Markdown(
                title="Confidence Distribution",
                is_header=True,
                formats=[
                    definitions.true_positives,
                    definitions.false_positives,
                ],
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget) -> Optional[go.Figure]:

        f1_optimal_conf, best_f1 = self._loader.mp.m_full.get_f1_optimal_conf()

        # Histogram of confidence scores (TP vs FP)
        scores_tp, scores_fp = self._loader.mp.m_full.scores_tp_and_fp()

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

        if f1_optimal_conf is not None:

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

        return fig


class PerClassAvgPrecision(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_class_ap=Widget.Markdown(
                title="Average Precision by Class",
                is_header=True,
                formats=[definitions.average_precision],
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget) -> Optional[go.Figure]:

        # AP per-class
        ap_per_class = self._loader.mp.coco_precision[:, :, :, 0, 2].mean(axis=(0, 1))
        # Per-class Average Precision (AP)
        fig = px.scatter_polar(
            r=ap_per_class,
            theta=self._loader.mp.cat_names,
            # title="Per-class Average Precision (AP)",
            labels=dict(r="Average Precision", theta="Category"),
            width=800,
            height=800,
            range_r=[0, 1],
        )
        # fill points
        fig.update_traces(fill="toself")

        return fig


class PerClassOutcomeCounts(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.clickable: bool = True
        self.switchable: bool = True
        self.schema = Schema(
            markdown_class_outcome_counts_1=Widget.Markdown(
                title="Outcome Counts by Class",
                is_header=True,
                formats=[
                    definitions.true_positives,
                    definitions.false_positives,
                    definitions.false_negatives,
                ],
            ),
            markdown_class_outcome_counts_2=Widget.Markdown(formats=[definitions.f1_score]),
            collapse_perclass_outcome=Widget.Collapse(
                Schema(markdown_normalization=Widget.Markdown(title="Normalization"))
            ),
            chart_01=Widget.Chart(switch_key="relative"),
            chart_02=Widget.Chart(switch_key="absolute"),
        )

    def get_figure(self, widget: Widget.Chart) -> Optional[go.Figure]:
        # Per-class Counts
        iou_thres = 0

        tp = self._loader.mp.true_positives[:, iou_thres]
        fp = self._loader.mp.false_positives[:, iou_thres]
        fn = self._loader.mp.false_negatives[:, iou_thres]

        # normalize
        support = tp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            tp_rel = tp / support
            fp_rel = fp / support
            fn_rel = fn / support

            # sort by f1
            sort_scores = 2 * tp / (2 * tp + fp + fn)

        K = len(self._loader.mp.cat_names)
        sort_indices = np.argsort(sort_scores)
        cat_names_sorted = [self._loader.mp.cat_names[i] for i in sort_indices]
        tp_rel, fn_rel, fp_rel = tp_rel[sort_indices], fn_rel[sort_indices], fp_rel[sort_indices]

        if widget.switch_key == "relative":
            # Stacked per-class counts
            data = {
                "count": np.concatenate([tp_rel, fn_rel, fp_rel]),
                "type": ["TP"] * K + ["FN"] * K + ["FP"] * K,
                "category": cat_names_sorted * 3,
            }
        elif widget.switch_key == "absolute":
            # Stacked per-class counts
            data = {
                "count": np.concatenate([tp[sort_indices], fn[sort_indices], fp[sort_indices]]),
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
        return fig

    def get_click_data(self, widget: Widget.Chart) -> Optional[dict]:
        if not self.clickable:
            return
        res = {}
        res["projectMeta"] = self._loader.dt_project_meta.to_json()
        res["clickData"] = {}
        for key1, v1 in self._loader.click_data.outcome_counts_by_class.items():
            for key2, v2 in v1.items():
                key = key1 + self._keypair_sep + key2
                res["clickData"][key] = {}
                res["clickData"][key]["layoutData"] = {}
                res["clickData"][key]["layout"] = []

                tmp = {0: [], 1: [], 2: [], 3: []}

                images = set(x["dt_img_id"] for x in v2)

                for idx, img_id in enumerate(images):
                    ui_id = f"ann_{img_id}"
                    info: ImageInfo = self._loader.dt_images_dct[img_id]
                    res["clickData"][key]["layoutData"][ui_id] = {
                        "imageUrl": info.preview_url,
                        "annotation": {
                            "imageId": info.id,
                            "imageName": info.name,
                            "createdAt": info.created_at,
                            "updatedAt": info.updated_at,
                            "link": info.link,
                            "annotation": self._loader.dt_ann_jsons[img_id],
                        },
                    }
                    if len(tmp[3]) < 5:
                        tmp[idx % 4].append(ui_id)

                for _, val in tmp.items():
                    res["clickData"][key]["layout"].append(val)

        return res


class OverallErrorAnalysis(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.cv_tasks: List[CVTask] = [CVTask.SEGMENTATION.value]

    def get_figure(self, widget: Widget.Chart) -> Optional[go.Figure]:
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

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.cv_tasks: List[CVTask] = [CVTask.SEGMENTATION.value]

    def get_figure(self, widget: Widget.Chart) -> Optional[go.Figure]:
        pd.options.mode.chained_assignment = None  # TODO rm later

        df = self._loader.result_df
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
