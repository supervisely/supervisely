import json
from pathlib import Path

from jinja2 import Template

from supervisely.io.fs import ensure_base_path
from supervisely.nn.benchmark.comparison.visualization.widgets.widget import BaseWidget


class ChartWidget(BaseWidget):
    def __init__(
        self,
        name: str,
        figure,  # plotly figure
        click_data: dict = None,
    ) -> None:
        super().__init__(name)
        self.radio_group = "radio_group"  # TODO: fix
        self.switch_key = "switch_key"  # TODO: fix
        self.init_data_source = f"/data/{self.name}_{self.id}.json"
        self.chart_click_data_source = f"/data/{self.name}_{self.id}_click_data.json"

        self.figure = figure
        self.click_data = click_data

    def save_data(self, basepath: str) -> None:
        # init data
        basepath = basepath.rstrip("/")
        ensure_base_path(basepath + self.init_data_source)

        with open(basepath + self.init_data_source, "w") as f:
            json.dump(self.get_init_data(), f)

        # click data
        if self.click_data is not None:
            ensure_base_path(basepath + self.chart_click_data_source)
            with open(basepath + self.chart_click_data_source, "w") as f:
                json.dump(self.click_data, f)

    def _get_template_data(self):
        return {
            "widget_id": self.id,
            "radio_group": self.radio_group,
            "switch_key": self.switch_key,
            "init_data_source": self.init_data_source,
            "chart_click_data_source": self.chart_click_data_source,
        }

    def to_html(self) -> str:
        Template(f"{Path(__file__).parent}/template.html").render()

    def get_init_data(self):
        return {
            "selected": None,
            "galleryContent": "",
            "dialogVisible": False,
            "chartContent": json.loads(self.figure.to_json()),
        }

    def save_state(self, basepath: str) -> None:
        return
