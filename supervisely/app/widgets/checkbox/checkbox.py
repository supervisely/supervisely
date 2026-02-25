from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

from supervisely.app import StateJson
from supervisely.app.widgets import Text, Widget


class Checkbox(Widget):
    """Checkbox for selecting an option."""

    class Routes:
        """Route name constants for this widget."""
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        content: Union[Widget, str],
        checked: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        """
        :param content: Label widget or text string.
        :type content: Union[Widget, str]
        :param checked: Initial checked state.
        :type checked: bool, optional
        :param widget_id: Widget identifier.
        :type widget_id: str, optional

        :Usage Example:

            .. code-block:: python

                from supervisely.app.widgets import Checkbox
                cb = Checkbox("Enable feature", checked=False)
        """
        self._content = content
        self._checked = checked
        if type(self._content) is str:
            self._content = [Text(self._content)][0]
        self._changes_handled = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        """Checkbox widget does not have any additional data, so it returns an empty dictionary.

        :returns: Empty dictionary.
        :rtype: Dict
        """
        return {}

    def get_json_state(self) -> Dict[str, bool]:
        """Returns the state of the checkbox.
        Dictionary contains the following keys:
            - checked: Current state of the checkbox.

        :returns: Dictionary with the state of the checkbox.
        :rtype: Dict[str, bool]
        """
        return {"checked": self._checked}

    def is_checked(self) -> bool:
        """Returns the state of the checkbox. True if checked, False otherwise.

        :returns: Current state of the checkbox.
        :rtype: bool
        """
        return StateJson()[self.widget_id]["checked"]

    def _set(self, checked: bool) -> None:
        """Sets the state of the checkbox.
        This is a private method, so it is not intended to be called directly.
        Use :meth:`check` or :meth:`uncheck` instead.

        :param checked: New state of the checkbox.
        :type checked: bool
        """
        self._checked = checked
        StateJson()[self.widget_id]["checked"] = self._checked
        StateJson().send_changes()

    def check(self) -> None:
        """Sets the state of the checkbox to True."""
        self._set(True)

    def uncheck(self) -> None:
        """Sets the state of the checkbox to False."""
        self._set(False)

    def value_changed(self, func: Callable[[bool], Any]) -> Callable[[], None]:
        """Decorator that handles the event of changing the state of the checkbox.

        :param func: Function that handles the event of changing the state of the checkbox.
        :type func: Callable[[bool], Any]
        :returns: Decorated function.
        :rtype: Callable[[], None]
        """
        route_path = self.get_route_path(Checkbox.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.is_checked()
            func(res)

        return _click
