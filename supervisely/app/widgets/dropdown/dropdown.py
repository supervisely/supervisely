from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Union

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class Dropdown(Widget):
    """Dropdown menu for selecting an action."""

    class Routes:
        """Route name constants for this widget."""
        VALUE_CHANGED = "value_changed"

    class Item:
        """Single dropdown menu item."""

        def __init__(
            self,
            text: Optional[str] = "",
            disabled: Optional[bool] = False,
            divided: Optional[bool] = False,
            command: Optional[Union[str, int]] = None,
        ) -> Dropdown.Item:
            """
            :param text: Item label.
            :param disabled: Disable item.
            :param divided: Show divider after item.
            :param command: Value emitted on click.
            """
            self.text = text
            self.disabled = disabled
            self.divided = divided
            self.command = command

        def to_json(self) -> Dict[str, Union[str, bool, int]]:
            """Return JSON representation of the item.

            Dictionary contains the following fields:
                - text: Item text
                - disabled: If True, item will be disabled
                - divided: If True, item will be divided from the next one
                - command: Item command

            :returns: Dictionary with item data
            :rtype: Dict[str, Union[str, bool, int]]
            """
            return {
                "text": self.text,
                "disabled": self.disabled,
                "divided": self.divided,
                "command": self.command,
            }

    def __init__(
        self,
        items: Optional[List[Dropdown.Item]] = None,
        header: Optional[str] = "Dropdown List",
        trigger: Optional[Literal["hover", "click"]] = "click",
        menu_align: Optional[Literal["start", "end"]] = "end",
        hide_on_click: Optional[bool] = True,
        widget_id: Optional[str] = None,
    ):
        """
        :param items: List of Dropdown.Item.
        :param header: Header text.
        :param trigger: "hover" or "click".
        :param menu_align: "start" or "end".
        :param hide_on_click: Close menu after selection.
        :param widget_id: Widget identifier.

        :Usage Example:

            .. code-block:: python

                from supervisely.app.widgets import Dropdown
                dd = Dropdown(items=[Dropdown.Item("Opt 1", command="a")], header="Options")
        """
        self._items = items
        self._header = header
        self._trigger = trigger
        self._menu_align = menu_align
        self._hide_on_click = hide_on_click
        self._changes_handled = False
        self._clicked_value = None

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _set_items(self):
        return [item.to_json() for item in self._items]

    def get_json_data(self) -> Dict[str, Any]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - trigger: Dropdown trigger type, one of: hover, click
            - items: List of items in the dropdown menu
            - menuAlign: Dropdown menu alignment, one of: start, end
            - hideOnClick: If True, dropdown menu will be hidden after click
            - header: Dropdown header text

        :returns: Dictionary with widget data
        :rtype: Dict[str, Union[str, Any]]
        """
        return {
            "trigger": self._trigger,
            "items": self._set_items(),
            "menuAlign": self._menu_align,
            "hideOnClick": self._hide_on_click,
            "header": self._header,
        }

    def get_json_state(self) -> Dict[str, str]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - clickedValue: current clicked value

        :returns: Dictionary with widget state
        :rtype: Dict[str, str]
        """
        return {"clickedValue": self._clicked_value}

    def get_value(self) -> str:
        """Returns current clicked value.

        :returns: current clicked value
        :rtype: str
        """
        return StateJson()[self.widget_id]["clickedValue"]

    def set_value(self, value: str) -> None:
        """Sets current clicked value.

        :param value: current clicked value
        :type value: str
        """
        self._clicked_value = value
        StateJson()[self.widget_id]["clickedValue"] = self._clicked_value
        StateJson().send_changes()

    def get_items(self) -> List[Dropdown.Item]:
        """Returns list of items in the dropdown menu.

        :returns: list of items in the dropdown menu
        :rtype: List[:class:`~supervisely.app.widgets.dropdown.dropdown.Dropdown.Item`]
        """
        return DataJson()[self.widget_id]["items"]

    def set_items(self, value: List[Dropdown.Item]) -> None:
        """Sets list of items in the dropdown menu.
        This method will overwrite all existing items.
        To add items to the dropdown menu, use :meth:`add_items` instead.

        :param value: list of items in the dropdown menu
        :type value: List[:class:`~supervisely.app.widgets.dropdown.dropdown.Dropdown.Item`]
        """
        if not all(isinstance(item, Dropdown.Item) for item in value):
            raise TypeError("Items must be a list of Dropdown.Item")
        self._items = value
        DataJson()[self.widget_id]["items"] = self._set_items()
        DataJson().send_changes()

    def add_items(self, value: List[Dropdown.Item]) -> None:
        """Adds items to the dropdown menu.
        This method will add items to the existing list.
        To overwrite all existing items, use :meth:`set_items` instead.

        :param value: list of items in the dropdown menu
        :type value: List[:class:`~supervisely.app.widgets.dropdown.dropdown.Dropdown.Item`]
        """
        self._items.extend(value)
        DataJson()[self.widget_id]["items"] = self._set_items()
        DataJson().send_changes()

    def get_header_text(self) -> str:
        """Returns dropdown header text.

        :returns: dropdown header text
        :rtype: str
        """
        return DataJson()[self.widget_id]["header"]

    def set_header_text(self, value: str) -> None:
        """Sets dropdown header text.

        :param value: dropdown header text
        :type value: str
        """
        if type(value) is not str:
            raise TypeError("Header value must be a string")
        self._header = value
        DataJson()[self.widget_id]["header"] = self._header
        DataJson().send_changes()

    def value_changed(self, func: Callable[[str], Any]) -> Callable[[], None]:
        """Decorator for the function to be called when the value is changed.

        :param func: function to be called when the value is changed
        :type func: Callable[[str], Any]
        :returns: decorated function
        :rtype: Callable[[], None]
        """
        route_path = self.get_route_path(Dropdown.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            func(res)

        return _click
