from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer

import plotly.graph_objects as go
from jinja2 import Template

import supervisely.nn.benchmark.visualization.vis_texts as contents
from supervisely._utils import camel_to_snake
from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.nn.benchmark.visualization.vis_templates import (
    template_chart_str,
    template_gallery_str,
    template_markdown_str,
    template_notification_str,
    template_radiogroup_str,
    template_table_str,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget


class MetricVis:

    def __init__(self, loader: Visualizer) -> None:

        self.cv_tasks: List[CVTask] = CVTask.values()
        self.clickable: bool = False
        self.has_diffs_view: bool = False
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
                    res += f"""\n          <div>\n            <el-button type="text" @click="data.scrollIntoView='{widget.id}'" """
                    res += (
                        """:style="{fontWeight: data.scrollIntoView==='"""
                        + widget.id
                        + """' ? 'bold' : 'normal'}"""
                    )
                    res += f""" ">{widget.title}</el-button>\n          </div>"""
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
                chart_click_path = f"/data/{basename}_click_data.json" if self.clickable else None
                chart_modal_data_source = f"/data/modal_general.json" if self.clickable else None
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
                        "chart_modal_data_source": chart_modal_data_source,
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
                gallery_modal_data_source = (
                    f"/data/{basename}_modal_data.json" if self.clickable else None
                )
                gallery_diff_data_source = (
                    f"/data/{basename}_diff_data.json" if self.has_diffs_view else None
                )
                res[f"{basename}_html"] = self._template_gallery.render(
                    {
                        "widget_id": widget.id,
                        "init_data_source": f"/data/{basename}.json",
                        "command": "command",
                        "data": "data",
                        "is_table_gallery": widget.is_table_gallery,
                        "gallery_click_data_source": gallery_click_data_source,
                        "gallery_diff_data_source": gallery_diff_data_source,
                        "gallery_modal_data_source": gallery_modal_data_source,
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
                        "table_gallery_id": f"modal_general",
                    }
                )

        return res

    @property
    def name(self) -> str:
        return camel_to_snake(self.__class__.__name__)

    def get_figure(self, widget: Widget.Chart) -> Optional[go.Figure]:
        pass

    def get_click_data(self, widget: Widget.Chart) -> Optional[dict]:
        if not self.clickable:
            return
        res = {}

        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}
        for key, v in self._loader.click_data.objects_by_class.items():
            res["clickData"][key] = {}
            res["clickData"][key]["imagesIds"] = []

            tmp = set()

            for x in v:
                dt_image = self._loader.dt_images_dct[x["dt_img_id"]]
                tmp.add(self._loader.diff_images_dct_by_name[dt_image.name].id)

            for img_id in tmp:
                res["clickData"][key]["imagesIds"].append(img_id)

            res["clickData"][key]["filters"] = [
                {"type": "tag", "tagId": "confidence", "value": [0.6, 1]},
                {"type": "tag", "tagId": "outcome", "value": "TP"},
            ]

        return res

    def get_modal_data(self, widget: Widget.Chart) -> Optional[dict]:
        res = {}
        api = self._loader._api
        gt_project_id = self._loader.gt_project_info.id
        dt_project_id = self._loader.dt_project_info.id
        diff_project_id = self._loader.diff_project_info.id
        gt_dataset = api.dataset.get_list(gt_project_id)[0]
        dt_dataset = api.dataset.get_list(dt_project_id)[0]
        diff_dataset = api.dataset.get_list(diff_project_id)[0]
        gt_image_infos = api.image.get_list(dataset_id=gt_dataset.id)[:3]
        pred_image_infos = api.image.get_list(dataset_id=dt_dataset.id)[:3]
        diff_image_infos = api.image.get_list(dataset_id=diff_dataset.id)[:3]
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
        res["projectMeta"] = project_metas[0].to_json()

        res.pop("layout")
        res.pop("layoutData")

        return res

    def get_table(self, widget: Widget.Table) -> Optional[dict]:
        pass

    def get_gallery(self, widget: Widget.Gallery) -> Optional[dict]:
        pass

    def get_gallery_click_data(self, widget: Widget.Gallery) -> Optional[dict]:
        pass

    def get_diff_gallery_data(self, widget: Widget.Gallery) -> Optional[dict]:
        pass

    def get_md_content(self, widget: Widget.Markdown):
        return getattr(contents, widget.name).format(*widget.formats)

    def initialize_formats(self, loader: Visualizer, widget: Widget):
        pass