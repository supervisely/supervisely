from datetime import datetime
from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import StateJson
from supervisely.app.widgets import Widget


class DatePicker(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        value: Union[int, str, list, tuple] = None,
        placeholder: str = "Select date",
        picker_type: Literal[
            "year", "month", "date", "datetime", "week", "datetimerange", "daterange"
        ] = "date",
        size: Literal["large", "small", "mini"] = None,
        readonly: bool = False,
        disabled: bool = False,
        editable: bool = True,
        clearable: bool = True,
        format: str = "yyyy-MM-dd",
        first_day_of_week: int = 1,
        widget_id: str = None,
    ):
        self._value = value
        self._readonly = readonly
        self._picker_type = picker_type
        self._size = size
        self._disabled = disabled
        self._placeholder = placeholder
        self._editable = editable
        self._clearable = clearable
        self._format = format
        self._first_day_of_week = first_day_of_week
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "type": self._picker_type,
            "size": self._size,
            "readonly": self._readonly,
            "disabled": self._disabled,
            "editable": self._editable,
            "clearable": self._clearable,
            "placeholder": self._placeholder,
            "format": self._format,
            "options": {"firstDayOfWeek": self._first_day_of_week},
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
        if self._picker_type in ["datetimerange", "daterange"] and any(
            [bool(date) is False for date in value]
        ):
            return None
        elif self._picker_type not in ["datetimerange", "daterange"] and value == "":
            return None
        return value

    def value_changed(self, func):
        route_path = self.get_route_path(DatePicker.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            res = self.get_value()
            func(res)

        return _value_changed

    def set_value(self, value: Union[int, str, datetime]):
        """
        Set your value to date picker.
        Date picker type mode is one of "date", "datetime", "year", "month" or "week".

        :type Union[int, str, datetime]
        - str, int, datetime
        """

        if self._picker_type in ["datetimerange", "daterange"]:
            raise ValueError(
                f'Date picker type "{self._picker_type}" is not abailable for this method. Try "set_range_values()"'
            )
        if type(value) not in [str, int, datetime]:
            raise ValueError(
                f'Value type {type(value)} is not matching for "{self._picker_type}" picker type.'
            )
        if isinstance(value, datetime):
            value = str(value)

        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def set_range_values(self, values: Union[list, tuple]):
        """
        Set your range values to date picker.
        Date picker type mode is one of "daterange" or "datetimerange".

        :type:
            Union[
                Tuple[str, str], Tuple[int, int], Tuple[datetime, datetime],
                List[str, str], List[int, int], List[datetime, datetime]
            ]
        """

        if self._picker_type not in ["datetimerange", "daterange"]:
            raise ValueError(
                f'Date picker type "{self._picker_type}" is not abailable for this method. Try "set_value()"'
            )
        if type(values) not in [list, tuple]:
            raise ValueError(
                f'Value type {type(values)} is not matching for "{self._picker_type}" picker type.'
            )
        if len(values) != 2:
            raise ValueError(f"Value length has to be equal 2: {len(values)} != 2")
        if any(isinstance(val, datetime) for val in values):
            values = [str(val) for val in values]

        self._value = values
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()
