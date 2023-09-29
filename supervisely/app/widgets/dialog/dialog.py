from typing import Literal, Optional

from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget


class Dialog(Widget):
    def __init__(
        self,
        title: Optional[str] = "",
        content: Optional[Widget] = None,
        size: Literal["tiny", "small", "large", "full"] = "small",
        widget_id: Optional[str] = None,
    ):
        self._title = title
        self._content = content
        self._size = size
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "title": self._title,
            "size": self._size,
        }

    def get_json_state(self):
        return {
            "visible": False,
        }

    def show(self):
        StateJson()[self.widget_id]["visible"] = True
        StateJson().send_changes()

    def hide(self):
        StateJson()[self.widget_id]["visible"] = False
        StateJson().send_changes()

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title
        DataJson()[self.widget_id]["title"] = title
        DataJson().send_changes()
