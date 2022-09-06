from __future__ import annotations

from typing import List
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class Menu(Widget):
    class Item:
        def __init__(
            self,
            title: str,
            content: Widget = None,
            index: str = None,
            icon: str = None,
        ) -> Menu.Item:
            self.title = title
            self.index = index
            self.content = content
            if index is None:
                self.index = str(self.title)
            self.icon = icon

        def to_json(self):
            return {"title": self.title, "index": self.index, "icon": self.icon}

    class Group:
        def __init__(self, title: str, items: List[Menu.Item] = None) -> Menu.Group:
            self.title = title
            self.items = items

        def to_json(self):
            return {
                "title": self.title,
                "items": [item.to_json() for item in self.items],
            }

    def __init__(
        self,
        items: List[Menu.Item] = None,
        groups: List[Menu.Group] = None,
        width_percent: int = 25,
        widget_id: str = None,
    ):

        self._groups = groups
        self._width_percent = width_percent
        self._options = {"sidebarWidth": self._width_percent}
        StateJson()["app_body_padding"] = "0px"
        StateJson()["menuIndex"] = "1"
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {"groups": [g.to_json() for g in self._groups]}

    def get_json_state(self):
        return {"menuIndex": self._groups[0].items[0].index}
