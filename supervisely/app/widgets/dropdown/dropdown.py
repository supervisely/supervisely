from __future__ import annotations
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Dropdown(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item:
        def __init__(
            self,
            text: str = "",
            disabled: bool = False,
            divided: bool = False,
            command: Union[str, int] = None,
        ) -> Dropdown.Item:
            self.text = text
            self.disabled = disabled
            self.divided = divided
            self.command = command

        def to_json(self):
            return {
                "text": self.text,
                "disabled": self.disabled,
                "divided": self.divided,
                "command": self.command,
            }

    def __init__(
        self,
        items: List[Dropdown.Item] = None,
        trigger: Literal["hover", "click"] = "hover",
        menu_align: Literal["start", "end"] = "end",
        hide_on_click: bool = True,
        header: str = "Dropdown List",
        widget_id: str = None,
    ):
        self._trigger = trigger
        self._items = items
        self._menu_align = menu_align
        self._hide_on_click = hide_on_click
        self._header = header
        self._changes_handled = False
        self._clicked_value = None

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _set_items(self):
        return [item.to_json() for item in self._items]

    def get_json_data(self):
        return {
            "trigger": self._trigger,
            "items": self._set_items(),
            "menu_align": self._menu_align,
            "hide_on_click": self._hide_on_click,
            "header": self._header,
        }

    def get_json_state(self):
        return {"clicked_value": self._clicked_value}

    def get_clicked_value(self):
        return StateJson()[self.widget_id]["clicked_value"]

    def set_clicked_value(self, value: str):
        self._clicked_value = value
        StateJson()[self.widget_id]["clicked_value"] = self._clicked_value
        StateJson().send_changes()

    def get_items(self):
        return DataJson()[self.widget_id]["items"]

    def set_items(self, value: List[Dropdown.Item]):
        self._items = value
        DataJson()[self.widget_id]["items"] = self._set_items()
        DataJson().send_changes()

    def add_items(self, value: List[Dropdown.Item]):
        self._items.extend(value)
        DataJson()[self.widget_id]["items"] = self._set_items()
        DataJson().send_changes()

    def expand_to_hover(self):
        self._trigger = "hover"
        DataJson()[self.widget_id]["trigger"] = self._trigger
        DataJson().send_changes()

    def expand_to_click(self):
        self._trigger = "click"
        DataJson()[self.widget_id]["trigger"] = self._trigger
        DataJson().send_changes()

    def set_menu_align_from_start(self):
        self._menu_align = "start"
        DataJson()[self.widget_id]["menu_align"] = self._menu_align
        DataJson().send_changes()

    def set_menu_align_from_end(self):
        self._menu_align = "end"
        DataJson()[self.widget_id]["menu_align"] = self._menu_align
        DataJson().send_changes()

    def unable_hide_on_click(self):
        self._hide_on_click = True
        DataJson()[self.widget_id]["hide_on_click"] = self._hide_on_click
        DataJson().send_changes()

    def disable_hide_on_click(self):
        self._hide_on_click = False
        DataJson()[self.widget_id]["hide_on_click"] = self._hide_on_click
        DataJson().send_changes()

    def get_header_text(self):
        return DataJson()[self.widget_id]["header"]

    def set_header_text(self, value: str):
        self._header = value
        DataJson()[self.widget_id]["header"] = self._header
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(Dropdown.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_clicked_value()
            func(res)

        return _click
