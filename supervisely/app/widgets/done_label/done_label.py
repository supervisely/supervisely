from typing import Literal
import markupsafe

from supervisely.app import DataJson
from supervisely.app.widgets import Widget

INFO = "info"
WARNING = "warning"
ERROR = "error"

from pathlib import Path
from jinja2 import Environment
import jinja2


class DoneLabel(Widget):
    def __init__(
        self,
        text: str = None,
        widget_id: str = None,
    ):
        self.text = text
        super().__init__(widget_id=widget_id, file_path=__file__)

    def init_data(self):
        return {"text": self.text}

    def init_state(self):
        return None
