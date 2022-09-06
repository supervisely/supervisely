from __future__ import annotations

from typing import List
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class SidebarMenu(Widget):
    # class Item:
    #     def __init__(self, value, label: str = None) -> SidebarMenu.Item:
    #         self.value = value
    #         self.label = label
    #         if label is None:
    #             self.label = str(self.value)

    #     def to_json(self):
    #         return {"label": self.label, "value": self.value}

    # class Group:
    #     def __init__(
    #         self, label, items: List[SidebarMenu.Item] = None
    #     ) -> SidebarMenu.Item:
    #         self.label = label
    #         self.items = items

    #     def to_json(self):
    #         res = {
    #             "label": self.label,
    #             "options": [item.to_json() for item in self.items],
    #         }
    #         return res

    def __init__(
        self,
        width_percent: int = 25,
        widget_id: str = None,
    ):
        super().__init__(widget_id=widget_id, file_path=__file__)
        self._left_pane = left_pane
        self._right_pane = right_pane
        self._width_percent = width_percent
        self._options = {"sidebarWidth": self._width_percent}
        StateJson()["app_body_padding"] = "0px"
        StateJson()["menuIndex"] = "1"

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}
