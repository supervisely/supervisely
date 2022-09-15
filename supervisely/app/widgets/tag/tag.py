from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Tag(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
            self,
            value: str = "",
            type: Literal["primary", "gray", "success", "warning", "danger"] = None,
            closable: bool = False,
            close_transition: bool = False,
            hit: bool = False,
            color: str = None,
            widget_id: str = None
    ):
        self._value = value
        self._type = type
        self._closable = closable
        self._close_transition = close_transition
        self._hit = hit
        self._color = color
        self._changes_handled = False
        self._widget_id = None

        super().__init__(widget_id=widget_id, file_path=__file__)

    # def to_json(self):
    #     return {
    #         "name": self._value,
    #         "type": self._type,
    #         "closable": self._closable,
    #         "closeTransition": self._close_transition,
    #         "hit": self._hit,
    #         "color": self._widget_id
    #     }

    def get_json_data(self):
        return {
            "type": self._type,
            "closable": self._closable,
            "closeTransition": self._close_transition,
            "hit": self._hit,
            "color": self._widget_id,
        }

    def get_json_state(self):
        return {"value": self._value}

    def set_value(self, value: str):
        StateJson()[self.widget_id]["value"] = value
        StateJson().send_changes()

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

    def set_type(self, value: str):
        if value not in ["primary", "gray", "success", "warning", "danger"]:
            return
        StateJson()[self.widget_id]["value"] = value
        StateJson().send_changes()

    def get_type(self):
        return StateJson()[self.widget_id]["value"]

    def is_closable(self):
        return StateJson()[self.widget_id]["closable"]

    def enable_closable(self):
        StateJson()[self.widget_id]["closable"] = True
        StateJson().send_changes()

    def disable_closable(self):
        StateJson()[self.widget_id]["closable"] = False
        StateJson().send_changes()

    def is_highlighted(self):
        return StateJson()[self.widget_id]["hit"]

    def enable_borderhighlighting(self):
        StateJson()[self.widget_id]["hit"] = True
        StateJson().send_changes()

    def disable_borderhighlighting(self):
        StateJson()[self.widget_id]["hit"] = False
        StateJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(Tag.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True
        @server.post(route_path)
        def _click():
            res = self.is_closable()
            print(res)
            func(res)
        return _click