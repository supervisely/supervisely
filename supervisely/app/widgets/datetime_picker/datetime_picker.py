from datetime import datetime
from typing import Any, Callable, Dict, Literal, Optional, Union

from supervisely.app import StateJson
from supervisely.app.widgets import Widget


class DateTimePicker(Widget):
    """DateTimePicker is a widget in Supervisely that allows you to choose a date and time on the UI.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/input/datetimepicker>`_
        (including screenshots and examples).

    :param value: initial value
    :type value: Optional[Union[int, str, datetime, list, tuple]]
    :param readonly: if True, date picker will be readonly
    :type readonly: Optional[bool]
    :param disabled: if True, date picker will be disabled
    :type disabled: Optional[bool]
    :param editable: if True, date picker will be editable
    :type editable: Optional[bool]
    :param clearable: if True, date picker will be clearable
    :type clearable: Optional[bool]
    :param size: size of the input box, one of: large, small, mini
    :type size: Optional[Literal["large", "small", "mini"]]
    :param placeholder: placeholder text, default: "Select date and time"
    :type placeholder: Optional[str]
    :param w_type: picker type, one of: year, month, date, datetime, week, datetimerange, daterange, default: datetime
    :type w_type: Optional[Literal["year", "month", "date", "datetime", "week", "datetimerange", "daterange"]]
    :param format: date format, one of: yyyy, MM, dd, HH, mm, ss, default: yyyy-MM-dd HH:mm:ss
    :type format: Optional[Literal["yyyy", "MM", "dd", "HH", "mm", "ss"]]
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

            from supervisely.app.widgets import DateTimePicker

            date_time_picker = DateTimePicker(
                value=datetime.now(),
                readonly=False,
                disabled=False,
                editable=False,
                clearable=True,
                size="small",
                placeholder="Select date and time",
                w_type="datetime",
                format="yyyy-MM-dd HH:mm:ss",
            )
    """

    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        value: Optional[Union[int, str, list, tuple]] = None,
        readonly: Optional[bool] = False,
        disabled: Optional[bool] = False,
        editable: Optional[bool] = False,
        clearable: Optional[bool] = True,
        size: Optional[Literal["large", "small", "mini"]] = None,
        placeholder: Optional[str] = "Select date and time",
        w_type: Optional[
            Literal["year", "month", "date", "datetime", "week", "datetimerange", "daterange"]
        ] = "datetime",
        format: Optional[Literal["yyyy", "MM", "dd", "HH", "mm", "ss"]] = "yyyy-MM-dd HH:mm:ss",
        widget_id: Optional[str] = None,
    ):
        self._value = value
        self._readonly = readonly
        self._disabled = disabled
        self._editable = editable
        self._clearable = clearable
        self._size = size
        self._placeholder = placeholder
        self._w_type = w_type
        self._format = format
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict[str, Union[str, bool]]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - placeholder: placeholder text
            - size: size of the input box, one of: large, small, mini
            - readonly: if True, date picker will be readonly
            - disabled: if True, date picker will be disabled
            - editable: if True, date picker will be editable
            - clearable: if True, date picker will be clearable
            - type: picker type, one of: year, month, date, datetime, week, datetimerange, daterange
            - format: date format, one of: yyyy, MM, dd, HH, mm, ss

        :return: dictionary with widget data
        :rtype: Dict[str, Union[str, bool]]
        """
        return {
            "placeholder": self._placeholder,
            "size": self._size,
            "readonly": self._readonly,
            "disabled": self._disabled,
            "editable": self._editable,
            "clearable": self._clearable,
            "type": self._w_type,
            "format": self._format,
        }

    def get_json_state(self) -> Dict[str, Union[str, list]]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - value: current value

        :return: dictionary with widget state
        :rtype: Dict[str, Union[str, list]]
        """
        return {"value": self._value}

    def get_value(self) -> Union[int, str, datetime, list, tuple]:
        """Returns current value.

        :return: current value
        :rtype: Union[int, str, datetime, list, tuple]
        """
        if "value" not in StateJson()[self.widget_id].keys():
            return None
        value = StateJson()[self.widget_id]["value"]
        if self._w_type in ["datetimerange", "daterange"] and any(
            [bool(date) is False for date in value]
        ):
            return None
        elif self._w_type not in ["datetimerange", "daterange"] and value == "":
            return None
        return value

    def set_value(self, value: Union[int, str, datetime, list, tuple]) -> None:
        """Sets current value.

        :param value: current value
        :type value: Union[int, str, datetime, list, tuple]
        :raises ValueError: if value type is not supported
        :raises ValueError: if value length is not equal 2
        """
        if self._w_type in ["year", "month", "date", "datetime", "week"]:
            if type(value) not in [int, str, datetime]:
                raise ValueError(
                    f'Datetime picker type "{self._w_type}" does not support value "{value}" of type: '
                    f'"{str(type(value))}". Value type has to be one of: ["int", "str", "datetime].'
                )
            if isinstance(value, datetime):
                value = str(value)

        if self._w_type in ["datetimerange", "daterange"]:
            if type(value) not in [list, tuple]:
                raise ValueError(
                    f'Datetime picker type "{self._w_type}" does not support value "{value}" of type: '
                    f'"{str(type(value))}". Value type has to be one of: ["list", "tuple"].'
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
        """Decorator for the function that will be called when the value of the widget is changed.

        :param func: function that will be called when the value of the widget is changed
        :type func: Callable[[Union[int, str, datetime, list, tuple]], Any]
        :return: decorated function
        :rtype: Callable[[], None]
        """
        route_path = self.get_route_path(DateTimePicker.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            self._value = res
            func(res)

        return _click
