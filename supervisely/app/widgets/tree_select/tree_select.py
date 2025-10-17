"""This module contains the implementation of the TreeSelect widget in Supervisely."""

# Source documentation: https://vue-treeselect.js.org/

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget


class TreeSelect(Widget):
    """TreeSelect widget in Supervisely is a widget that allows users to select items from a tree-like structure.
    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/selection/treeselect>`_
        (including screenshots and examples).

    :param items: List of items to display in the tree.
    :type items: List[TreeSelect.Item], optional
    :param multiple_select: Whether multiple items can be selected at once.
    :type multiple_select: bool, optional
    :param flat: If set to true, selecting a parent item will NOT select all its children.
    :type flat: bool, optional
    :param always_open: If set to true, the widget will be expanded by default.
    :type always_open: bool, optional
    :param width: The width of the widget.
    :type width: int, optional
    :param append_to_body: Determines where the popover is attached. If False, it is positioned inside the input's container. This can cause the popover to be hidden if the input is within a Card or a widget that restricts visibility.
    :type append_to_body: bool, optional
    :widget_id: The unique identifier of the widget.
    :type widget_id: str, optional
    :param placeholder: The placeholder text.

    :Public methods:

    - `get_selected() -> Union[List[TreeSelect.Item], TreeSelect.Item]`: Get the selected item(s).
    - `set_selected(value: Union[List[TreeSelect.Item], TreeSelect.Item])`: Set the selected item(s).
    - `set_items(items: List[TreeSelect.Item])`: Set the items (overwrite the existing items).
    - `add_items(items: List[TreeSelect.Item])`: Add the items (append to the existing items).
    - `clear_items()`: Clear the items.
    - `get_item_by_id(item_id: str) -> Optional[TreeSelect.Item]`: Get the item by its ID.
    - `set_selected_by_id(value: Union[List[str], str])`: Set the selected item(s) by their IDs.
    - `clear_selected()`: Clear the selected item(s).
    - `get_all_items() -> List[TreeSelect.Item]`: Get all items in the tree.
    - `select_all()`: Select all items, including children.
    - `is_all_selected() -> bool`: Check if all items are selected.

    :Usage example:

        .. code-block:: python
            from supervisely.app.widgets import TreeSelect

            items = [
                TreeSelect.Item(id="1", label="First item", children=[
                    TreeSelect.Item(id="1.1", label="First child"),
                    TreeSelect.Item(id="1.2", label="Second child"),
                ]),
                TreeSelect.Item(id="2", label="Second item"),
            ]

            # Initialize the widget without items.
            tree_select = TreeSelect(multiple_select=True, flat=True, always_open=True)

            # Set the items.
            tree_select.set_items(items)
    """

    class Routes:
        """Endpoints for the widget."""

        VALUE_CHANGED = "value_changed"

    class Item:
        """Class representing an item in the tree.

        :param id: The unique identifier of the item.
        :type id: str
        :param label: The label of the item.
        :type label: str, optional
        :param children: The children of the item.
        :type children: List[TreeSelect.Item], optional

        :Usage example:

            .. code-block:: python

                item = TreeSelect.Item(id="1", label="First item", children=[
                    TreeSelect.Item(id="1.1", label="First child"),
                    TreeSelect.Item(id="1.2", label="Second child"),
                ])
        """

        def __init__(
            self,
            id: str,
            label: Optional[str] = None,
            children: List[TreeSelect.Item] = None,
        ):
            self.id = id
            self.label = label or id
            self.children = children or []

        def to_json(self) -> Dict[str, Union[str, List[Dict]]]:
            data = {
                "id": self.id,
                "label": self.label,
            }
            if self.children:
                data["children"] = [child.to_json() for child in self.children]

            return data

        @classmethod
        def from_json(cls, data: Dict[str, Union[str, List[Dict]]]) -> TreeSelect.Item:
            return cls(
                id=data["id"],
                label=data.get("label"),
                children=[TreeSelect.Item.from_json(child) for child in data.get("children", [])],
            )

        def __repr__(self):
            return f"Item(id={self.id}, label={self.label}, children={self.children})"

    def __init__(
        self,
        items: Optional[List[TreeSelect.Item]] = None,
        multiple_select: bool = False,
        flat: bool = False,
        always_open: bool = False,
        width: Optional[int] = None,
        append_to_body: bool = True,
        widget_id: Optional[str] = None,
        placeholder: Optional[str] = None,
        show_tooltip: bool = True,
    ):
        self._items = items or []
        self._multiple = multiple_select
        self._flat = flat
        self._always_open = always_open
        self._value_format = "object"  # On the frontend side can be "object" or "id".
        self._value = [] if multiple_select else None
        self._width = width
        self._append_to_body = append_to_body
        self._placeholder = placeholder
        self._show_tooltip = show_tooltip

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict[str, List[Dict]]:
        """Return the JSON representation of the widget data.

        :return: The JSON representation of the widget data.
        :rtype: Dict[str, List[Dict]]
        """
        return {
            "items": [item.to_json() for item in self._items] if self._items else [],
            "width": self._width,
        }

    def get_json_state(self) -> Dict[str, Union[Dict, List[Dict]]]:
        """Return the JSON representation of the widget state.

        :return: The JSON representation of the widget state.
        :rtype: Dict[str, Union[Dict, List[Dict]]]
        """
        return {
            "value": self.value,
            "options": {
                "multiple": self._multiple,
                "flat": self._flat,
                "alwaysOpen": self._always_open,
                "valueFormat": self._value_format,
                "appendToBody": self._append_to_body,
                "placeholder": self._placeholder,
                "showTooltip": self._show_tooltip,
            },
        }

    @property
    def value(self) -> Union[List[TreeSelect.Item], TreeSelect.Item]:
        """Return the selected item(s).
        Do not use this property directly, use get_selected() instead.

        :return: The selected item(s).
        :rtype: Union[List[TreeSelect.Item], TreeSelect.Item]
        """
        return self._value

    @value.setter
    def value(self, value: Union[List[TreeSelect.Item], TreeSelect.Item]):
        """Set the selected item(s).
        Do not use this property directly, use set_selected() instead.

        :param value: The selected item(s).
        :type value: Union[List[TreeSelect.Item], TreeSelect.Item]
        """
        self._value = value

    def _get_value(self) -> Union[List[TreeSelect.Item], TreeSelect.Item]:
        """Get the selected item as instances of the Item class.

        :return: The selected item(s).
        :rtype: Union[List[TreeSelect.Item], TreeSelect.Item]
        """
        res = StateJson()[self.widget_id]["value"]
        if res is None:
            return None
        if isinstance(res, list):
            return [TreeSelect.Item.from_json(item) for item in res]
        return TreeSelect.Item.from_json(res)

    def _set_value(self, value: Optional[Union[List[TreeSelect.Item], TreeSelect.Item]]):
        """Set the selected item(s) as instances of the Item class.

        :param value: The selected item(s).
        :type value: Union[List[TreeSelect.Item], TreeSelect.Item]
        """
        self.value = value
        if isinstance(value, list):
            json_value = [item.to_json() for item in value]
        elif value is not None:
            json_value = value.to_json()
        else:
            json_value = None if not self._multiple else []
        StateJson()[self.widget_id]["value"] = json_value
        StateJson().send_changes()

    def clear_selected(self) -> None:
        """Clear the selected item(s)."""
        self._set_value([] if self._multiple else None)

    def get_selected(self) -> Union[List[TreeSelect.Item], TreeSelect.Item]:
        """Get the selected item(s).

        :return: The selected item(s).
        :rtype: Union[List[TreeSelect.Item], TreeSelect.Item]
        """
        return self._get_value()

    def set_selected(self, value: Union[List[TreeSelect.Item], TreeSelect.Item]):
        """Set the selected item(s).

        :param value: The selected item(s).
        :type value: Union[List[TreeSelect.Item], TreeSelect.Item]
        :raises ValueError: If the widget is set to single selection mode and a list of items
            is provided.
        """
        if self._multiple and not isinstance(value, list):
            value = [value]
        if not self._multiple and isinstance(value, list):
            raise ValueError(
                "The widget is set to single selection mode, but a list of items was provided."
                "Either set the widget to multiple selection mode or provide a single item."
            )

        self._set_value(value)

    def get_all_items(self) -> List[TreeSelect.Item]:
        """Get all items in the tree.

        :return: All items in the tree.
        :rtype: List[TreeSelect.Item]
        """

        def _get_all_items(items: List[TreeSelect.Item]) -> List[TreeSelect.Item]:
            res = []
            if not items:
                return res
            for item in items:
                res.append(item)
                res.extend(_get_all_items(item.children))
            return res

        return _get_all_items(self._items)

    def select_all(self) -> None:
        """Select all items, including children."""
        if not self._multiple:
            raise ValueError(
                "The widget is set to single selection mode, but tried to select all items."
            )
        self._set_value(self.get_all_items())

    def is_all_selected(self) -> bool:
        """Check if all items are selected.

        :return: True if all items are selected, False otherwise.
        :rtype: bool
        """
        if not self._multiple:
            raise ValueError(
                "The widget is set to single selection mode, but tried to check if all items are selected."
            )

        all_item_ids = {item.id for item in self.get_all_items()}
        selected_ids = {item.id for item in self.get_selected()}

        return all_item_ids.issubset(selected_ids)

    def get_item_by_id(self, item_id: str) -> Optional[TreeSelect.Item]:
        """Get the item by its ID.

        :param item_id: The ID of the item.
        :type item_id: str
        :return: The item with the specified ID.
        :rtype: Optional[TreeSelect.Item]
        """

        def _get_item_by_id(
            items: List[TreeSelect.Item], item_id: str
        ) -> Optional[TreeSelect.Item]:
            for item in items:
                if item.id == item_id:
                    return item
                res = _get_item_by_id(item.children, item_id)
                if res:
                    return res
            return None

        return _get_item_by_id(self._items, item_id)

    def set_selected_by_id(self, value: Union[List[str], str]) -> None:
        if not self._multiple and isinstance(value, list):
            raise ValueError(
                "The widget is set to single selection mode, but a list of items was provided."
                "Either set the widget to multiple selection mode or provide a single item."
            )
        if not isinstance(value, list):
            value = [value]

        items = []
        for item_id in value:
            item = self.get_item_by_id(item_id)
            if item:
                items.append(item)

        if items:
            self.set_selected(items if self._multiple else items[0])

    def _update_items(
        self, items: Union[List[TreeSelect.Item], TreeSelect.Item], overwrite: bool
    ) -> None:
        """Update the items in the widget.

        :param items: The items to update.
        :type items: Union[List[TreeSelect.Item], TreeSelect.Item]
        :param overwrite: Whether to overwrite the existing items.
        :type overwrite: bool
        """
        if overwrite:
            self._items = items
            self.clear_selected()
        else:
            self._items.extend(items)
        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()

    def set_items(self, items: List[TreeSelect.Item]) -> None:
        """Set the items in the widget.

        :param items: The items to set.
        :type items: List[TreeSelect.Item]
        """
        self._update_items(items, overwrite=True)

    def add_items(self, items: List[TreeSelect.Item]) -> None:
        """Add the items to the widget.

        :param items: The items to add.
        :type items: List[TreeSelect.Item]
        """
        self._update_items(items, overwrite=False)

    def clear_items(self) -> None:
        """Clear the items in the widget."""
        items = [] if self._multiple else None
        self._update_items(items, overwrite=True)

    def value_changed(self, func: Callable) -> Callable:
        """Decorator for the function to be called when the value is changed.

        :param func: The function to be called when the value is changed.
        :type func: Callable
        :return: The decorated function.
        :rtype: Callable
        """
        route_path = self.get_route_path(TreeSelect.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self._get_value()

            func(res)

        return _click
