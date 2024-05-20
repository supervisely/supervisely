from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

from supervisely.app import StateJson
from supervisely.app.widgets import Text, Widget


class CheckboxField(Widget):
    """This widget is a simple and intuitive interface element that allows users to select given option.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/controls/checkbox>`_
        (including screenshots and examples).

    :param content: Content of the checkbox.
    :type content: Union[Widget, str]
    :param checked: Initial state of the checkbox.
    :type checked: Optional[bool]
    :param widget_id: Unique widget identifier.
    :type widget_id: str
    """

    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        title: str,
        description: str,
        checked: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        self._title = title
        self._description = description
        self._checked = checked
        self._changes_handled = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        """Checkbox widget does not have any additional data, so it returns an empty dictionary.

        :return: Empty dictionary.
        :rtype: Dict
        """
        return {}

    def get_json_state(self) -> Dict[str, bool]:
        """Returns the state of the checkbox.
        Dictionary contains the following keys:
            - checked: Current state of the checkbox.

        :return: Dictionary with the state of the checkbox.
        :rtype: Dict[str, bool]
        """
        return {"checked": self._checked, "title": self._title, "description": self._description}

    def is_checked(self) -> bool:
        """Returns the state of the checkbox. True if checked, False otherwise.

        :return: Current state of the checkbox.
        :rtype: bool
        """
        return StateJson()[self.widget_id]["checked"]

    def set(self, checked: bool = None, title: str = None, description: str = None) -> None:
        """Sets the state of the checkbox.
        This is a private method, so it is not intended to be called directly.
        Use :meth:`check` or :meth:`uncheck` instead.

        :param checked: New state of the checkbox.
        :type checked: bool
        """
        if checked is not None:
            self._checked = checked
            StateJson()[self.widget_id]["checked"] = self._checked

        if title is not None:
            self._title = title
            StateJson()[self.widget_id]["title"] = self._title

        if description is not None:
            self._description = description
            StateJson()[self.widget_id]["description"] = self._description
        self.update_state()
        StateJson().send_changes()

    def check(self) -> None:
        """Sets the state of the checkbox to True."""
        self._checked = True
        StateJson()[self.widget_id]["checked"] = self._checked
        StateJson().send_changes()

    def uncheck(self) -> None:
        """Sets the state of the checkbox to False."""
        self._checked = False
        StateJson()[self.widget_id]["checked"] = self._checked
        StateJson().send_changes()

    def value_changed(self, func: Callable[[bool], Any]) -> Callable[[], None]:
        """Decorator that handles the event of changing the state of the checkbox.

        :param func: Function that handles the event of changing the state of the checkbox.
        :type func: Callable[[bool], Any]
        :return: Decorated function.
        :rtype: Callable[[], None]
        """
        route_path = self.get_route_path(CheckboxField.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.is_checked()
            func(res)

        return _click
