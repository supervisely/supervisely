from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class TextArea(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        value: str = None,
        placeholder: str = "Please input",
        rows: int = 2,
        autosize: bool = True,
        readonly: bool = False,
        widget_id=None,
    ):
        self._value = value
        self._placeholder = placeholder
        self._rows = rows
        self._autosize = autosize
        self._readonly = readonly
        self._changes_handled = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            # "value": self._value,
            "placeholder": self._placeholder,
            "rows": self._rows,
            "autosize": self._autosize,
            "readonly": self._readonly,
        }

    def get_json_state(self):
        return {"value": self._value}

    def set_value(self, value):
        # self._value = value
        StateJson()[self.widget_id]["value"] = value
        StateJson().send_changes()

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

    def is_readonly(self):
        return DataJson()[self.widget_id]["readonly"]

    def enable_readonly(self):
        self._readonly = True
        DataJson()[self.widget_id]["readonly"] = True
        DataJson().send_changes()

    def disable_readonly(self):
        self._readonly = False
        DataJson()[self.widget_id]["readonly"] = False
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(TextArea.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            self._value = res
            func(res)

        return _click
