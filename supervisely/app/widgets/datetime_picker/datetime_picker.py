from supervisely.app import StateJson
from supervisely.app.widgets import Widget
from datetime import datetime

try:
    from typing import Literal, Union
except ImportError:
    from typing_extensions import Literal


class DateTimePicker(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        value: Union[int, str, list, tuple] = None,
        readonly: bool = False,
        disabled: bool = False,
        editable: bool = True,
        clearable: bool = True,
        size: Literal["large", "small", "mini"] = None,
        placeholder: str = "Select date and time",
        type: Literal[
            "year", "month", "date", "datetime", "week", "datetimerange", "daterange"
        ] = "datetime",
        format: str = "yyyy-MM-dd HH:mm:ss",
        widget_id: str = None,
    ):
        self._value = value
        self._readonly = readonly
        self._disabled = disabled
        self._editable = editable
        self._clearable = clearable
        self._size = size
        self._placeholder = placeholder
        self._type = type
        self._format = format
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "placeholder": self._placeholder,
            "size": self._size,
            "readonly": self._readonly,
            "disabled": self._disabled,
            "editable": self._editable,
            "clearable": self._clearable,
            "type": self._type,
            "format": self._format,
        }

    def get_json_state(self):
        return {"value": self._value}

    def clear_value(self):
        self._value = None
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_value(self):
        if "value" not in StateJson()[self.widget_id].keys():
            return None
        value = StateJson()[self.widget_id]["value"]
        if self._type in ["datetimerange", "daterange"] and any(
            [bool(date) is False for date in value]
        ):
            return None
        elif self._type not in ["datetimerange", "daterange"] and value == "":
            return None
        return value

    def set_value(self, value: Union[int, str, datetime]):
        """
        Set your value to datetime picker.
        This method supports one of "date", "datetime", "year", "month" or "week" mode.
        :type Union[int, str, datetime]
        - str, int, datetime
        """

        if self._type in ["datetimerange", "daterange"]:
            raise ValueError(
                f'Datetime picker type "{self._type}" is not abailable for this method. Try "set_range_values()"'
            )
        if type(value) not in [str, int, datetime]:
            raise ValueError(
                f'Value type {type(value)} is not matching for "{self._type}" picker type.'
            )
        if isinstance(value, datetime):
            value = str(value)

        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def set_range_values(self, values: Union[list, tuple]):
        """
        Set your range values to datetime picker.
        This method supports one of "daterange" or "datetimerange" mode.
        :type:
            Union[
                Tuple[str, str], Tuple[int, int], Tuple[datetime, datetime],
                List[str, str], List[int, int], List[datetime, datetime]
            ]
        """

        if self._type not in ["datetimerange", "daterange"]:
            raise ValueError(
                f'Datetime picker type "{self._type}" is not abailable for this method. Try "set_value()"'
            )
        if type(values) not in [list, tuple]:
            raise ValueError(
                f'Value type {type(values)} is not matching for "{self._type}" picker type.'
            )
        if len(values) != 2:
            raise ValueError(f"Value length has to be equal 2: {len(values)} != 2")
        if any(isinstance(val, datetime) for val in values):
            values = [str(val) for val in values]

        self._value = values
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(DateTimePicker.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            self._value = res
            func(res)

        return _click
