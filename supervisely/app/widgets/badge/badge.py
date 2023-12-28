from typing import Dict, Optional, Union

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class Badge(Widget):
    """Badge widget in Supervisely is a versatile tool for displaying notifications or counts on elements such as buttons, text.
    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/status-elements/badge>`_
        (including screenshots and examples).

    :param value: Value to be displayed on the badge.
    :type value: Optional[Union[int, str, float]]
    :param widget: Widget to be displayed on the badge.
    :type widget: Optional[Widget]
    :param max: Maximum value of the badge. If value is greater than max, max will be displayed on the badge.
    :type max: Optional[Union[int, float]]
    :param is_dot: If True, the badge will be displayed as a dot.
    :type is_dot: Optional[bool]
    :param hidden: If True, the badge will be hidden.
    :type hidden: Optional[bool]
    :param widget_id: Unique widget identifier.
    :type widget_id: Optional[str]

    :Usage example:
    .. code-block:: python
        from supervisely.app.widgets import Badge

        badge = Badge(value=5, max=10)
    """

    def __init__(
        self,
        value: Optional[Union[int, str, float]] = None,
        widget: Optional[Widget] = None,
        max: Optional[Union[int, float]] = None,
        is_dot: Optional[bool] = False,
        hidden: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        self._value = value
        self._widget = widget
        self._max = max if type(max) in [int, float] else None
        self._hidden = hidden
        self._is_dot = is_dot

        if self._value is None and self._widget is not None:
            self._is_dot = True

        if self._is_dot is True and self._value is None:
            self._value = 0

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict[str, Union[int, float, bool]]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - max: Maximum value of the badge. If value is greater than max, max will be displayed on the badge.
            - isDot: If True, the badge will be displayed as a dot.
            - hidden: If True, the badge will be hidden.

        :return: Dictionary with widget data.
        :rtype: Dict[str, Union[int, float, bool]]
        """
        res = {}
        res["max"] = self._max
        res["isDot"] = self._is_dot
        res["hidden"] = self._hidden
        return res

    def get_json_state(self) -> Dict[str, Union[str, int, float]]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - value: Value to be displayed on the badge.

        :return: Dictionary with widget state.
        :rtype: Dict[str, Union[str, int, float]]
        """
        return {"value": self._value}

    def set_value(self, value: Union[str, int, float]) -> None:
        """Sets value to be displayed on the badge.

        :param value: Value to be displayed on the badge.
        :type value: Union[str, int, float]
        """
        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_value(self) -> Union[str, int, float]:
        """Returns value to be displayed on the badge.

        :return: Value to be displayed on the badge.
        :rtype: Union[str, int, float]
        """
        if "value" not in StateJson()[self.widget_id].keys():
            return None
        value = StateJson()[self.widget_id]["value"]
        return value

    def clear(self) -> None:
        """Clears the value of the badge."""
        self._value = None
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def hide_badge(self) -> None:
        """Hides the badge."""
        self._hidden = True
        DataJson()[self.widget_id]["hidden"] = self._hidden
        DataJson().send_changes()

    def show_badge(self) -> None:
        """Shows the badge."""
        self._hidden = False
        DataJson()[self.widget_id]["hidden"] = self._hidden
        DataJson().send_changes()

    def toggle_visibility(self) -> None:
        """Toggles the visibility of the badge.
        If the badge is hidden, it will be shown.
        If the badge is shown, it will be hidden.
        """
        self._hidden = not self._hidden
        DataJson()[self.widget_id]["hidden"] = self._hidden
        DataJson().send_changes()
