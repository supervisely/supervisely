from __future__ import annotations
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Dict, Union

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
        widget_id: str = None,
    ):
        self._trigger = trigger
        self._items = items
        self._menu_align = menu_align
        self._hide_on_click = hide_on_click
        self._changes_handled = False
        self._clicked_command = None

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _set_items(self):
        return [item.to_json() for item in self._items]

    def get_json_data(self):
        return {
            "trigger": self._trigger,
            "items": self._set_items(),
            "menu_align": self._menu_align,
            "hide_on_click": self._hide_on_click,
        }

    def get_json_state(self):
        return {"command": self._clicked_command}

    def get_value(self):
        return StateJson()[self.widget_id]["command"]

    def value_changed(self, func):
        route_path = self.get_route_path(Dropdown.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            func(res)

        return _click
