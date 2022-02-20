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


class NotificationBox(Widget):
    def __init__(
        self,
        title: str = None,
        description: str = None,
        box_type: Literal["info", "warning", "error"] = WARNING,
        widget_id: str = None,
    ):
        self.title = title
        self.description = description
        if self.title is None and self.description is None:
            raise ValueError(
                "Both title and description can not be None at the same time"
            )
        self.box_type = box_type
        self.icon = "zmdi-alert-triangle"  # @TODO: get by box type
        if self.box_type != WARNING:
            raise ValueError(
                f"Only {WARNING} type is supported. Other types {[INFO, WARNING, ERROR]} will be supported later"
            )

        super().__init__(widget_id=widget_id, file_path=__file__)

    def init_data(self):
        return {"title": self.title, "description": self.description, "icon": self.icon}

    def init_state(self):
        return None
