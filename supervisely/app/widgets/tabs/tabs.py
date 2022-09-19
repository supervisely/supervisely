from __future__ import annotations

from typing import List, Union
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget, Text

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Tabs(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class TabPane:
        def __init__(
                self,
                label: str = "",
                name: str = "",
                content: str = None
        ) -> Tabs.TabPane:
            self._label = label
            self._name = name  # identifier corresponding to the active tab
            self._content = content

        def to_json(self):
            return {
                "label": self._label,
                "name": self._name,
                "content": self._content
            }

    def __init__(
            self,
            items: List[TabPane] = [],
            value: str = None,
            type: Literal["card", "border-card"] = None,
            widget_id=None
    ):
        self._items = items
        self._value = value
        self._type = type
        self._value_changed = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "items": self._items,
            "value": self._value,
            "type": self._type
        }

    def get_json_state(self):
        return {}

    def set_active_tab(self, value):
        DataJson()[self.widget_id]["value"] = value
        DataJson().send_changes()

    def get_active_tab(self):
        return DataJson()[self.widget_id]["value"]
