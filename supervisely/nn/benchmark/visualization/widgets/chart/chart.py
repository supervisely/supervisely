import json
from pathlib import Path
from typing import Dict

from jinja2 import Template

from supervisely.io.fs import ensure_base_path
from supervisely.nn.benchmark.visualization.widgets.widget import BaseWidget


class ChartWidget(BaseWidget):
    def __init__(
        self,
        name: str,
        figure,  # plotly figure
        switchable: bool = False,
        switch_key: str = "switch_key",
        radiogroup_id: str = None,
    ) -> None:
        super().__init__(name)
        self.switchable = switchable
        self.switch_key = switch_key
        self.radiogroup_id = radiogroup_id

        self.figure = figure
        self.click_data = None
        self.click_gallery_id = None
        self.chart_click_extra = None

    def save_data(self, basepath: str) -> None:
        # init data
        basepath = basepath.rstrip("/")
        ensure_base_path(basepath + self.data_source)

        with open(basepath + self.data_source, "w") as f:
            json.dump(self.get_init_data(), f)

        # click data
        if self.click_data is not None:
            ensure_base_path(basepath + self.click_data_source)
            with open(basepath + self.click_data_source, "w") as f:
                json.dump(self.click_data, f)

    def set_click_data(
        self, gallery_id: str, click_data: Dict, chart_click_extra: str = ""
    ) -> None:
        self.click_data = click_data
        self.click_gallery_id = gallery_id
        self.chart_click_extra = chart_click_extra

    def _get_template_data(self):
        return {
            "widget_id": self.id,
            "radio_group": self.radiogroup_id,
            "switchable": self.switchable,
            "switch_key": self.switch_key,
            "init_data_source": self.data_source,
            "click_handled": self.click_data is not None,
            "chart_click_data_source": self.click_data_source,
            "gallery_id": self.click_gallery_id,
            "chart_click_extra": self.chart_click_extra,
        }

    def to_html(self) -> str:
        template_str = Path(__file__).parent / "template.html"
        return Template(template_str.read_text()).render(self._get_template_data())

    def get_init_data(self):
        return {
            "selected": None,
            "galleryContent": "",
            "dialogVisible": False,
            "chartContent": json.loads(self.figure.to_json()),
        }

    def get_state(self) -> Dict:
        res = {}
        if self.switchable:
            res[self.radiogroup_id] = self.switch_key
        return res
