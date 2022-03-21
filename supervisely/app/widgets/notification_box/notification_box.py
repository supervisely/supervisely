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
        self._title = title
        self._description = description
        if self._title is None and self._description is None:
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

    def get_serialized_data(self):
        return {"title": self._title, "description": self._description, "icon": self.icon}

    def get_serialized_state(self):
        return None

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value
        self.update_data(data=DataJson())

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self.update_data(data=DataJson())



