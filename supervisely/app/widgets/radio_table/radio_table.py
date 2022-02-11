from typing import List, Dict
from pathlib import Path
import os
import markupsafe
from jinja2 import Environment
from supervisely.app.jinja2 import create_env

# from supervisely.api.api import Api
# from supervisely.app.content import DataJson, LastStateJson


class RadioTable:
    def __init__(self, state_field: str, data_field: str, content: List[Dict] = None):
        self.state_field = state_field
        self.data_field = data_field
        self.content = content

    def to_html(self):
        current_dir = Path(__file__).parent.absolute()
        jinja2_sly_env: Environment = create_env(current_dir)
        html = jinja2_sly_env.get_template("radio_table.html").render({"widget": self})
        return markupsafe.Markup(html)
