from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Text, Widget


class CheckboxField(Widget):
    """This widget is a checkbox with a description.

    :param title: Title of the checkbox.
    :type title: str
    :param description: Description of the checkbox.
    :type description: str
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
        """Returns the data of the checkbox.
        Dictionary contains the following keys:
            - title: Current title data of the checkbox.
            - description: Current description data of the checkbox

        :return: Dictionary with the data of the checkbox.
        :rtype: Dict[str, str]
        """
        return {"title": self._title, "description": self._description}

    def get_json_state(self) -> Dict[str, bool]:
        """Returns the state of the checkbox.
        Dictionary contains the following keys:
            - checked: Current state of the checkbox.

        :return: Dictionary with the state of the checkbox.
        :rtype: Dict[str, bool]
        """
        return {"checked": self._checked}

    def is_checked(self) -> bool:
        """Returns the state of the checkbox. True if checked, False otherwise.

        :return: Current state of the checkbox.
        :rtype: bool
        """
        return StateJson()[self.widget_id]["checked"]

    def set(self, title: str = None, description: str = None, checked: bool = None) -> None:
        """Set title, description and checkbox state to widget

        :param title: Title of the checkbox.
        :type title: Optional[str]
        :param description: Description of the checkbox.
        :type description: Optional[str]
        :param checked: Initial state of the checkbox.
        :type checked: Optional[bool]
        :param checked: New state of the checkbox.
        :type checked: bool
        """
        if checked is not None:
            self._checked = checked
            StateJson()[self.widget_id]["checked"] = self._checked
            StateJson().send_changes()
        if title is not None:
            self._title = title
            DataJson()[self.widget_id]["title"] = self._title
        if description is not None:
            self._description = description
            DataJson()[self.widget_id]["description"] = self._description
        if title is not None or description is not None:
            DataJson().send_changes()

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
