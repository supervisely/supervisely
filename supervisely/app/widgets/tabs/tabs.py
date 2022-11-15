from __future__ import annotations

from typing import List, Union, Optional
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget, Text
import uuid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Tabs(Widget):
    class TabPane:
        def __init__(self, label: Optional[str] = "", content: Optional[Widget] = None):
            self.label = label
            self.name = label  # identifier corresponding to the active tab
            self.content = content

    def __init__(
        self,
        labels: List[str],
        contents: List[Widget],
        type: Literal["card", "border-card"] = "border-card",
        widget_id=None,
    ):
        if len(labels) != len(contents):
            raise ValueError(
                "items_labels length must be equal to items_content length in Tabs widget."
            )
        if len(set(labels)) != len(labels):
            raise ValueError("All of tab labels should be unique.")
        self._items = []
        for label, widget in zip(labels, contents):
            self._items.append(Tabs.TabPane(label=label, content=widget))
        self._value = labels[0]
        self._type = type
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {"type": self._type}

    def get_json_state(self):
        return {"value": self._value}

    def set_active_tab(self, value):
        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_active_tab(self):
        return StateJson()[self.widget_id]["value"]
