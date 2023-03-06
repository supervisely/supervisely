from __future__ import annotations
from supervisely.app import StateJson
from supervisely.app.widgets import Widget, Text
from typing import List, Dict, Union


class Checkbox(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        content: Union[Widget, str],
        checked: bool = False,
        disabled: bool = False,
        widget_id: str = None,
    ):
        self._content = content
        self._checked = checked
        self._disabled = disabled
        if type(self._content) is str:
            self._content = [Text(self._content)][0]
        self._changes_handled = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {"disabled": self._disabled}

    def get_json_state(self) -> Dict:
        return {
            "checked": self._checked,
            "disabled": self._disabled
        }

    def is_checked(self):
        return StateJson()[self.widget_id]["checked"]

    def _set(self, checked: bool):
        self._checked = checked
        self._disabled = disabled
        StateJson()[self.widget_id]["checked"] = self._checked
        StateJson()[self.widget_id]["disabled"] = self._disabled
        StateJson().send_changes()
        self._set_disabled()

    def _set_disabled(self):
        self._content.set_attr("disabled", self._disabled)

    def check(self):
        self._set(True)

    def uncheck(self):
        self._set(False)

    def set_disabled(self, disabled: bool):  # New method.
        self._disabled = disabled
        self._set_disabled()

    def value_changed(self, func):
        route_path = self.get_route_path(Checkbox.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.is_checked()
            func(res)

        return _click