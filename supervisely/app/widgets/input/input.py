from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Input(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        value: str = "",
        minlength: int = 0,
        maxlength: int = 1000,
        placeholder: str = "",
        size: Literal["mini", "small", "large"] = None,
        readonly: bool = False,
        type: Literal["text", "password", "textarea"] = "text",
        widget_id: str = None,
        icon: Literal["search", "edit"] = None,
        autosize_textarea: bool = True,
    ):
        self._value = value  # initial value
        self._minlength = minlength
        self._maxlength = maxlength
        self._placeholder = placeholder
        self._size = size
        self._readonly = readonly
        self._widget_id = widget_id
        self._changes_handled = False
        self._type = type
        self._icon = icon
        self._autosize_textarea = autosize_textarea

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "minlength": self._minlength,
            "maxlength": self._maxlength,
            "placeholder": self._placeholder,
            "size": self._size,
            "readonly": self._readonly,
            "autosize": self._autosize_textarea,
        }

    def get_json_state(self):
        return {
            "value": self._value,
            "type": self._type,
            "icon": self._icon,
        }

    @property
    def autosize_textarea(self):
        return self._autosize_textarea

    @autosize_textarea.setter
    def autosize_textarea(self, value: bool):
        self._autosize_textarea = value
        DataJson()[self.widget_id]["autosize_textarea"] = value

    def set_value(self, value):
        StateJson()[self.widget_id]["value"] = value
        StateJson().send_changes()

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

    def set_type(self, value: Literal["text", "password"]):
        StateJson()[self.widget_id]["type"] = value
        StateJson().send_changes()

    def get_type(self):
        return StateJson()[self.widget_id]["type"]

    def set_icon(self, value: Literal["search", "edit"]):
        StateJson()[self.widget_id]["icon"] = value
        StateJson().send_changes()

    def get_icon(self):
        return StateJson()[self.widget_id]["icon"]

    def is_readonly(self):
        return DataJson()[self.widget_id]["readonly"]

    def enable_readonly(self):
        DataJson()[self.widget_id]["readonly"] = True
        DataJson().send_changes()

    def disable_readonly(self):
        DataJson()[self.widget_id]["readonly"] = False
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(Input.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            # self._value = res  # commented cause we do not need to update initial value
            func(res)

        return _click
