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

        self.figure = figure
        self.click_data = click_data

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

    def _get_template_data(self):
        return {
            "widget_id": self.id,
            "radio_group": self.radio_group,
            "switch_key": self.switch_key,
            "init_data_source": self.data_source,
            "chart_click_data_source": self.click_data_source,
        }

    def to_html(self) -> str:
        template_str = Path(__file__).parent / "template.html"
        return Template(template_str.read_text()).render()

    def get_init_data(self):
        return {
            "selected": None,
            "galleryContent": "",
            "dialogVisible": False,
            "chartContent": json.loads(self.figure.to_json()),
        }

    def save_state(self, basepath: str) -> None:
        return
