from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Cascader(Widget):
    """Cascader is a dropdown list with hierarchical options.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/selection/cascader>`_
        (including screenshots and examples).

    :param items: List of Cascader.Item objects to be displayed in the cascader.
    :type items: Optional[List[Cascader.Item]]
    :param selected_items: List of selected items.
    :type selected_items: Optional[List[str]]
    :param filterable: If True, the cascader will be filterable.
    :type filterable: Optional[bool]
    :param placeholder: Placeholder text of the cascader.
    :type placeholder: Optional[str]
    :param size: Size of the cascader.
    :type size: Optional[Literal["large", "small", "mini"]]
    :param expand_trigger: Trigger type to expand the cascader.
    :type expand_trigger: Optional[Literal["click", "hover"]]
    :param clearable: If True, the cascader will be clearable.
    :type clearable: Optional[bool]
    :param show_all_levels: If True, all levels of the cascader will be displayed.
    :type show_all_levels: Optional[bool]
    :param parent_selectable: If True, parent items will be selectable.
    :type parent_selectable: Optional[bool]
    :param widget_id: Unique widget identifier.
    :type widget_id: str

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import Cascader

        cascader_items = [
            Cascader.Item(value="cat", label="Cat", children=[
                Cascader.Item(value="black cat", label="Black Cat"),
                Cascader.Item(value="fluffy cat", label="Fluffy Cat"),
            ]),
            Cascader.Item(value="dog", label="Dog", children=[
                Cascader.Item(value="black dog", label="Black Dog"),
                Cascader.Item(value="fluffy dog", label="Fluffy Dog"),
            ]),
        ]

        cascader = Cascader(
            items=cascader_items, selected_items=[], filterable=True, placeholder="select",
            size="small", expand_trigger="click", clearable=True, show_all_levels=True,
            parent_selectable=False
            )
    """

    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item:
        """Represents an item in the cascader.

        :param value: Value of the item.
        :type value: str
        :param label: Label of the item.
        :type label: Optional[str]
        :param children: Children of the item.
        :type children: Optional[List[Cascader.Item]]
        :param disabled: If True, the item will be disabled.
        :type disabled: Optional[bool]
        """

        def __init__(
            self,
            value: str,
            label: Optional[str] = None,
            children: Optional[List[Cascader.Item]] = [],
            disabled: Optional[bool] = False,
        ) -> Cascader.Item:
            self.value = value
            self.label = label
            if label is None:
                self.label = str(self.value)
            self.children = children
            self.disabled = disabled

        def to_json(self) -> Dict[str, Union[str, bool, List[Cascader.Item]]]:
            """Returns dictionary with item data.

            Dictionary contains the following fields:
                - value: Value of the item.
                - label: Label of the item.
                - children: Children of the item.
                - disabled: If True, the item will be disabled.
            """
            children = []
            for child in self.children:
                children.append(child.to_json())
            if len(children) == 0:
                return {"label": self.label, "value": self.value, "disabled": self.disabled}
            else:
                return {
                    "label": self.label,
                    "value": self.value,
                    "disabled": self.disabled,
                    "children": children,
                }

    def __init__(
        self,
        items: Optional[List[Cascader.Item]] = None,
        selected_items: Optional[List[str]] = None,
        filterable: Optional[bool] = False,
        placeholder: Optional[str] = "select",
        size: Optional[Literal["large", "small", "mini"]] = None,
        expand_trigger: Optional[Literal["click", "hover"]] = "click",
        clearable: Optional[bool] = True,
        show_all_levels: Optional[bool] = True,
        parent_selectable: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        self._items = items
        self._selected_items = selected_items
        self._filterable = filterable
        self._placeholder = placeholder
        self._size = size
        self._expand_trigger = expand_trigger
        self._clearable = clearable
        self._show_all_levels = show_all_levels
        self._parent_selectable = parent_selectable
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _to_json(self, items: List[Cascader.Item]):
        return [item.to_json() for item in items]

    def _set_items(self):
        DataJson()[self.widget_id]["items"] = self._to_json(self._items)
        DataJson().send_changes()

    def get_json_data(self) -> Dict[str, Any]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - filterable: If True, the cascader will be filterable.
            - placeholder: Placeholder text of the cascader.
            - expandTrigger: Trigger type to expand the cascader.
            - size: Size of the cascader.
            - disabled: If True, the cascader will be disabled.
            - clearable: If True, the cascader will be clearable.
            - showAllLevels: If True, all levels of the cascader will be displayed.
            - parentSelectable: If True, parent items will be selectable.
            - items: List of cascader items.

        :return: Dictionary with widget data.
        :rtype: Dict[str, Any]
        """
        res = {
            "filterable": self._filterable,
            "placeholder": self._placeholder,
            "expandTrigger": self._expand_trigger,
            "size": self._size,
            "disabled": self._disabled,
            "clearable": self._clearable,
            "showAllLevels": self._show_all_levels,
            "parentSelectable": self._parent_selectable,
        }
        if self._items is not None:
            res["items"] = self._to_json(self._items)

        return res

    def get_json_state(self) -> Dict[str, List[Cascader.Item]]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - selectedItems: List of selected items.

        :return: Dictionary with widget state.
        :rtype: Dict[str, List[Cascader.Item]]
        """
        return {"selectedItems": self._selected_items}

    def get_selected_items(self) -> List[str]:
        """Returns list of values of selected items.

        :return: List of values of selected items.
        :rtype: List[str]
        """
        return StateJson()[self.widget_id]["selectedItems"]

    def select_item(self, values: List[Union[str, Cascader.Item]]):
        """Selects item by value.

        Can be used to select multiple items, can be passed as a list of values or a list of Cascader.Item objects.

        :param values: List of values of items to be selected.
        :type values: List[Union[str, Cascader.Item]]]

        :Usage example:
        .. code-block:: python


            # Selects item by string value
            cascader.select_item(["cat", "black cat", "fluffy cat"])
            # Selects item by Cascader.Item object
            cascader.select_item([Cascader.Item("cat"), Cascader.Item("black cat"), Cascader.Item("fluffy cat")])
        """

        str_values = []
        for item in values:
            if isinstance(item, Cascader.Item):
                str_values.append(item.value)
            else:
                str_values.append(item)

        if len(str_values) > 0:
            last_item = str_values[-1]
            last_item = self.get_item(last_item)
            if last_item is not None:
                if len(last_item.children) > 0:
                    # raise ValueError("Selected item must not have children")
                    str_values = []

        self._selected_items = str_values
        StateJson()[self.widget_id]["selectedItems"] = self._selected_items
        StateJson().send_changes()

    def deselect(self) -> None:
        """Deselects all items."""
        self.select_item([])

    def get_items(self) -> List[Cascader.Item]:
        """Returns list of Cascader.Item objects.

        :return: List of Cascader.Item objects.
        :rtype: List[Cascader.Item]
        """
        return DataJson()[self.widget_id]["items"]

    def get_item(self, value: str) -> Cascader.Item:
        """Returns Cascader.Item object by value.

        :param value: Value of the item.
        :type value: str
        :return: Cascader.Item object.
        :rtype: Cascader.Item
        """

        def _recursive_search(items: List[Cascader.Item]) -> Cascader.Item:
            """Recursively searches for item by its value.

            :param items: List of Cascader.Item objects.
            :type items: List[Cascader.Item]
            :return: Cascader.Item object.
            :rtype: Cascader.Item
            """
            for item in items:
                if item.value == value:
                    return item
                found = _recursive_search(item.children)
                if found is not None:
                    return found
            return None

        return _recursive_search(self._items)

    def set_items(self, items: List[Cascader.Item]) -> None:
        """Sets list of Cascader.Item objects to be displayed in the cascader.

        This method will overwrite the existing items, not append to it.
        To append items, use :meth:`add_items`.

        :param items: List of Cascader.Item objects to be displayed in the cascader.
        :type items: List[Cascader.Item]
        :raises TypeError: If items are not of type Cascader.Item.
        """
        if any(not isinstance(item, Cascader.Item) for item in items):
            raise TypeError("All items must be of type Cascader.Item")
        self._items = items
        self._set_items()
        self.deselect()

    def add_item(self, item: Cascader.Item) -> None:
        """Appends Cascader.Item object to the existing items.

        This method will not overwrite the existing items, but append to it.
        To overwrite items, use :meth:`set_items`.

        :param item: Cascader.Item object to be displayed in the cascader.
        :type item: Cascader.Item
        """
        self.add_items([item])

    def add_items(self, items: List[Cascader.Item]) -> None:
        """Appends list of Cascader.Item objects to the existing items.

        This method will not overwrite the existing items, but append to it.
        To overwrite items, use :meth:`set_items`.

        :param items: List of Cascader.Item objects to be displayed in the cascader.
        :type items: List[Cascader.Item]
        :raises TypeError: If items are not of type Cascader.Item.
        :raises ValueError: If item with the same value already exists.
        """
        if any(not isinstance(item, Cascader.Item) for item in items):
            raise TypeError("All items must be of type Cascader.Item")
        for item in items:
            if item.value in [i.value for i in self._items]:
                raise ValueError(f"Item with value '{item.value}' already exists.")
        self._items.extend(items)
        self._set_items()

    def value_changed(self, func: Callable[[List[str]], Any]) -> Callable[[], None]:
        """Decorator for the function to be called when the value of the cascader is changed.

        :param func: Function to be called when the value of the cascader is changed.
        :type func: Callable[[List[str]], Any]
        :return: Decorated function.
        :rtype: Callable[[], None]
        """
        route_path = self.get_route_path(Cascader.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_selected_items()
            func(res)

        return _click
