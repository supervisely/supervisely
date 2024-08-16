from __future__ import annotations

from typing import Iterator, Optional

from jinja2 import Template

import supervisely.nn.benchmark.visualization.vis_texts as contents
from supervisely._utils import camel_to_snake, rand_str


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

            from supervisely.app.widgets.grid_gallery_v2.grid_gallery_v2 import (
                GridGalleryV2,
            )

            self.gallery = GridGalleryV2(
                columns_number=3,
                annotations_opacity=0.5,
                border_width=4,
                enable_zoom=False,
                default_tag_filters=[{"confidence": [0.6, 1]}, {"outcome": "TP"}],
                show_zoom_slider=False,
            )

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
                annotations_opacity=0.5,
                border_width=4,
                enable_zoom=False,
                default_tag_filters=[{"confidence": [0.6, 1]}, {"outcome": "FP"}],
                show_zoom_slider=False,
            )

            super().__init__()