from typing import Any, Dict, Optional, Tuple, Union

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class BindedInputNumber(Widget):
    """BindedInputNumber widget in Supervisely is a user interface element that allows users
    to input two numerical values and customize their behavior using the proportional, min, and max properties.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/input/bindedinputnumber>`_
        (including screenshots and examples).

    :param width: Width of the image in pixels.
    :type width: Optional[int]
    :param height: Height of the image in pixels.
    :type height: Optional[int]
    :param min: Minimum value of the input.
    :type min: Optional[int]
    :param max: Maximum value of the input.
    :type max: Optional[int]
    :param proportional: If True, the input will be proportional.
    :type proportional: Optional[bool]
    :param widget_id: Unique widget identifier.
    :type widget_id: str

    :Usage example:
    .. code-block:: python
        from supervisely.app.widgets import Badge

        binded_input_number = BindedInputNumber(width=150, height=150, min=1, max=100, proportional=False)
    """

    def __init__(
        self,
        width: Optional[int] = 256,
        height: Optional[int] = 256,
        min: Optional[int] = 1,
        max: Optional[int] = 10000,
        proportional: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        self._width = width
        self._height = height
        self._min = min
        self._max = max
        self._proportional = proportional
        self._disabled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict[str, Any]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - options: Dictionary with the following fields:
                - proportions: Dictionary with the following fields:
                    - width: Width of the image in pixels.
                    - height: Height of the image in pixels.
            - disabled: If True, the widget will be disabled.
        """
        return {
            "options": {"proportions": {"width": self._width, "height": self._height}},
            "disabled": self._disabled,
        }

    def get_json_state(self) -> Dict[str, Union[int, bool]]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
               - value: Dictionary with the following fields:
                   - min: Minimum value of the input.
                   - max: Maximum value of the input.
                   - width: Width of the image in pixels.
                   - height: Height of the image in pixels.
                   - proportional: If True, the input will be proportional.
        """
        return {
            "value": {
                "min": self._min,
                "max": self._max,
                "width": self._width,
                "height": self._height,
                "proportional": self._proportional,
            }
        }

    @property
    def value(self) -> Tuple[int, int]:
        """Returns the width and height of the image.

        :return: Tuple with the width and height of the image.
        :rtype: Tuple[int, int]
        """
        return self._width, self._height

    @value.setter
    def value(self, width: int, height: int) -> None:
        """Sets the width and height of the image.

        :param width: Width of the image in pixels.
        :type width: int
        :param height: Height of the image in pixels.
        :type height: int
        """
        self._width = width
        self._height = height
        StateJson()[self.widget_id]["width"] = self._width
        StateJson()[self.widget_id]["height"] = self._height
        StateJson().send_changes()

    def get_value(self) -> Tuple[int, int]:
        """Returns the width and height of the image.

        :return: Tuple with the width and height of the image.
        :rtype: Tuple[int, int]
        """
        width = StateJson()[self.widget_id]["value"]["width"]
        height = StateJson()[self.widget_id]["value"]["height"]
        return width, height

    @property
    def proportional(self) -> bool:
        """Returns True if the input is proportional, False otherwise."""
        return self._proportional

    @property
    def min(self) -> int:
        """Returns the minimum value of the input."""
        return self._min

    @property
    def max(self) -> int:
        """Returns the maximum value of the input."""
        return self._max

    @proportional.setter
    def proportional(self, value) -> None:
        """Sets the proportional value.

        :param value: If True, the input will be proportional.
        :type value: bool

        :Usage example:
        .. code-block:: python
            binded_input_number.proportional = True
        """
        self._proportional = value
        DataJson()[self.widget_id]["proportional"] = self._proportional
        DataJson().send_changes()

    @min.setter
    def min(self, value) -> None:
        """Sets the minimum value of the input.

        :param value: Minimum value of the input.
        :type value: int
        """
        self._min = value
        DataJson()[self.widget_id]["min"] = self._min
        DataJson().send_changes()

    @max.setter
    def max(self, value) -> None:
        """Sets the maximum value of the input.

        :param value: Maximum value of the input.
        :type value: int
        """
        self._max = value
        DataJson()[self.widget_id]["max"] = self._max
        DataJson().send_changes()

    def disable(self) -> None:
        """Disables the widget in the UI."""
        self._disabled = True
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self) -> None:
        """Enables the widget in the UI."""
        self._disabled = False
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()
