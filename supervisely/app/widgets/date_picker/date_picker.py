from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from supervisely.app import StateJson
from supervisely.app.widgets import Widget


class DatePicker(Widget):
    """DatePicker widget in Supervsely is a user-friendly and customizable date input solution for Supervisely app developers.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/input/datepicker>`_
        (including screenshots and examples).

    :param value: initial value
    :type value: Optional[Union[int, str, datetime, list, tuple]]
    :param placeholder: placeholder text, default: "Select date"
    :type placeholder: Optional[str]
    :param picker_type: picker type, one of: year, month, date, datetime, week, datetimerange, daterange, default: date
    :type picker_type: Optional[Literal["year", "month", "date", "datetime", "week", "datetimerange", "daterange"]]
    :param size: size of the input box, one of: large, small, mini
    :type size: Optional[Literal["large", "small", "mini"]]
    :param readonly: if True, date picker will be readonly
    :type readonly: Optional[bool]
    :param disabled: if True, date picker will be disabled
    :type disabled: Optional[bool]
    :param editable: if True, date picker will be editable
    :type editable: Optional[bool]
    :param clearable: if True, date picker will be clearable
    :type clearable: Optional[bool]
    :param format: date format, one of: yyyy, MM, dd, HH, mm, ss, default: yyyy-MM-dd
    :type format: Optional[Literal["yyyy", "MM", "dd", "HH", "mm", "ss"]]
    :param first_day_of_week: first day of week, default: 1
    :type first_day_of_week: Optional[int]
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import DatePicker

        date_picker = DatePicker(
            value=datetime.now(),
            placeholder="Select date",
            picker_type="date",
            size="small",
            readonly=False,
            disabled=False,
            editable=False,
            clearable=True,
            format="yyyy-MM-dd",
            first_day_of_week=1,
        )
    """

    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        value: Optional[Union[int, str, list, tuple]] = None,
        placeholder: Optional[str] = "Select date",
        picker_type: Optional[
            Literal["year", "month", "date", "datetime", "week", "datetimerange", "daterange"]
        ] = "date",
        size: Optional[Literal["large", "small", "mini"]] = None,
        readonly: Optional[bool] = False,
        disabled: Optional[bool] = False,
        editable: Optional[bool] = False,
        clearable: Optional[bool] = True,
        format: Optional[Literal["yyyy", "MM", "dd", "HH", "mm", "ss"]] = "yyyy-MM-dd",
        first_day_of_week: Optional[int] = 1,
        widget_id: Optional[str] = None,
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

    def get_json_data(self) -> Dict[str, Union[str, bool, int, Dict[str, int]]]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - type: string representation of the widget type (year, month, date, datetime, week, datetimerange, daterange)
            - size: size of the input box, one of: large, small, mini
            - readonly: if True, date picker will be readonly
            - disabled: if True, date picker will be disabled
            - editable: if True, date picker will be editable
            - clearable: if True, date picker will be clearable
            - placeholder: placeholder text
            - format: date format, one of: yyyy, MM, dd, HH, mm, ss
            - options: dictionary with options for date picker
                - firstDayOfWeek: first day of week

        :return: dictionary with widget data
        :rtype: Dict[str, Union[str, bool, int, Dict[str, int]]]
        """

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

    def get_json_state(self) -> Dict[str, Union[str, List[int]]]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - value: current value

        :return: dictionary with widget state
        :rtype: Dict[str, Union[str, List[int]]]
        """
        return {"value": self._value}

    def clear_value(self) -> None:
        """Clears current value of the widget."""
        self._value = None
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_value(self) -> Union[int, str, datetime, list, tuple, None]:
        """Returns current value.

        :return: current value
        :rtype: Union[int, str, datetime, list, tuple, None]
        """
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

    def set_value(self, value: Union[int, str, datetime, list, tuple]) -> None:
        """Sets current value.

        :param value: current value
        :type value: Union[int, str, datetime, list, tuple]
        """
        if self._picker_type in ["year", "month", "date", "datetime", "week"]:
            if type(value) not in [int, str, datetime]:
                raise ValueError(
                    f'Date picker type "{self._picker_type}" does not support value "{value}" of type: "{str(type(value))}". Value type has to be one of: ["int", "str", "datetime].'
                )
            if isinstance(value, datetime):
                value = str(value)

        if self._picker_type in ["datetimerange", "daterange"]:
            if type(value) not in [list, tuple]:
                raise ValueError(
                    f'Date picker type "{self._picker_type}" does not support value "{value}" of type: "{str(type(value))}". Value type has to be one of: ["list", "tuple"].'
                )
            if len(value) != 2:
                raise ValueError(f"Value length has to be equal 2: {len(value)} != 2")
            value = [str(val) if isinstance(val, datetime) else val for val in value]

        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def value_changed(
        self, func: Callable[[Union[int, str, datetime, list, tuple]], Any]
    ) -> Callable[[], None]:
        """Decorator fot the function that will be called when value is changed.

        :param func: function that will be called when value is changed
        :type func: Callable[[Union[int, str, datetime, list, tuple]], Any]
        :return: decorated function
        :rtype: Callable[[], None]
        """
        route_path = self.get_route_path(DatePicker.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            res = self.get_value()
            func(res)

        return _value_changed
