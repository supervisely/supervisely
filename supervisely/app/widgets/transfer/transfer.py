from __future__ import annotations
from collections import namedtuple
from supervisely.app.widgets import Widget
from supervisely.app.content import DataJson, StateJson

from typing import List, Dict, Union, Optional, Callable

class Transfer(Widget):
    """Widget for transferring items between source and target lists. Operates with a single list; use transferred_items for keys to show in target."""

    class Routes:
        """Route name constants for this widget."""
        VALUE_CHANGED = "value_changed"

    class Item:
        """Class for representing items in the Transfer widget."""

        def __init__(self, key: str, label: Optional[str] = None, disabled: Optional[bool] = False):
            """Initialize Transfer.Item.

            :param key: Unique key for identifying the item.
            :type key: str
            :param label: Label displayed in the widget. If None, key is used.
            :type label: Optional[str]
            :param disabled: If True, the item won't be transferable.
            :type disabled: Optional[bool]

            :Usage Example:

                .. code-block:: python

                    Transfer.Item(key="dog", label="Dog", disabled=True)
                    Transfer.Item(key="cat", label="Cat")
                    Transfer.Item(key="mouse")
            """
            self.key = key
            if not label:
                # If label is not specified, the key will be used as label.
                self.label = key
            else:
                self.label = label
            self.disabled = disabled

        def to_json(self):
            return {"key": self.key, "label": self.label, "disabled": self.disabled}

    def __init__(
        self,
        items: Optional[Union[List[Item], List[str]]] = None,
        transferred_items: Optional[List[str]] = None,
        widget_id: Optional[str] = None,
        filterable: Optional[bool] = False,
        filter_placeholder: Optional[str] = None,
        titles: Optional[List[str]] = None,
        button_texts: Optional[List[str]] = None,
        left_checked: Optional[List[str]] = None,
        right_checked: Optional[List[str]] = None,
        width: int = 150,
    ):
        """Initialize the Transfer widget.

        :param items: List of items or list of strings (keys, auto-transformed into items).
        :type items: Union[List[:class:`~supervisely.app.widgets.transfer.transfer.Transfer.Item`], List[str]], optional
        :param transferred_items: Keys of items to display in the right (target) list.
        :type transferred_items: List[str], optional
        :param widget_id: The id of the widget.
        :type widget_id: str, optional
        :param filterable: If True, the widget will have a filter input.
        :type filterable: bool, optional
        :param filter_placeholder: Placeholder for the filter input (only if filterable=True).
        :type filter_placeholder: str, optional
        :param titles: Titles for source and target lists. Default: ('Source', 'Target').
        :type titles: List[str], optional
        :param button_texts: Texts for transfer buttons. Default: icons only.
        :type button_texts: List[str], optional
        :param left_checked: Keys of items to check in left list at init.
        :type left_checked: List[str], optional
        :param right_checked: Keys of items to check in right list at init.
        :type right_checked: List[str], optional
        :param width: Width of the widget in pixels. Minimum 150.
        :type width: int, optional

        :Usage Example:

            .. code-block:: python

                from supervisely.app.widgets import Transfer

                item1 = Transfer.Item(key="cat", label="Cat", disabled=True)
                item2 = Transfer.Item(key="dog", label="Dog")
                transfer = Transfer(items=[item1, item2], transferred_items=["dog"])
                transfer.set_transferred_items(["cat"])
        """
        self._changes_handled = False
        self._items = []
        self._transferred_items = []

        if items:
            self._items = self.__checked_items(items)

        if transferred_items:
            self._transferred_items = self.__checked_transferred_items(transferred_items)

        # If wrong items are specified, items won't be checked.
        self._left_checked = left_checked
        self._right_checked = right_checked

        self._filterable = filterable
        self._filter_placeholder = filter_placeholder

        self._width = max(width, 150)

        self._titles = titles if titles is not None else ["Source", "Target"]

        self._button_texts = button_texts

        super().__init__(widget_id=widget_id, file_path=__file__)

    def __checked_items(self, items: Optional[Union[List[Item], List[str]]]) -> List[Transfer.Item]:
        """
        If the list of items is specified as a list of strings, they will be converted to items. List of
        Transfer items will be checked for uniqueness of the keys. If the keys of the items are not unique, an error will be
        raised.

        :param items: The list of items can either be a list of items or a list of strings, containing the
            keys for items to be created.
        :type items: Optional[Union[List[Item], List[str]]]

        :raises ValueError: If the keys of the items are not unique.

        :returns: The list of items with unique keys.
        :rtype: List[:class:`~supervisely.app.widgets.transfer.transfer.Transfer.Item`]
        """

        if isinstance(items[0], str):
            # If items are specified as strings, they will be converted to Transfer.Item objects.
            if len(set(items)) != len(items):
                # If the keys of the items are not unique, an error will be raised.
                raise ValueError("The keys of the items should be unique.")

            checked_items = [Transfer.Item(key=item) for item in items]
        else:
            # If items are specified as Transfer.Item objects, they will be checked for uniqueness.
            if len({item.key for item in items}) != len(items):
                # If the keys of the items are not unique, an error will be raised.
                raise ValueError("The keys of the items should be unique.")

            checked_items = items
        return checked_items

    def __checked_transferred_items(self, transferred_items: List[str]) -> List[str]:
        """
        If the self._items is specified, the list of transferred items will be checked for the keys of the items. Since
        transferred items are specified by keys of the items, each key of the transferred items should exist in the list of
        items. Otherwise, an error will be raised.

        :param transferred_items: List of keys of the items to be shown in the right (target) list.
        :type transferred_items: List[str]

        :raises ValueError: If transferred items are specified, but the list of items is not specified.
        :raises ValueError: If any of transferred items keys is not in the list of items.

        :returns: List of transferred items (keys of the items which should be displayed in the right list).
        :rtype: List[str]
        """

        if not self._items:
            # If the list of items is not specified, the list of transferred items can't be specified since
            # the keys of the items should be specified in the list of transferred items.
            raise ValueError("The 'items' argument should be specified if "\
                "the 'transferred_items' argument is specified.")
        else:
            if not set(transferred_items).issubset([item.key for item in self._items]):
                # If any of the keys of the items specified in the list of transferred items is not
                # in the list of items, an error will be raised.
                raise ValueError("The 'transferred_items' argument should contain only "\
                    "the keys of the items specified in the 'items' argument.")
            else:
                return transferred_items

    def get_json_data(self) -> Dict[str, Union[List[Dict[str, Union[str, bool]]], None]]:
        """
        Returns the data of the widget in JSON format.

        Data will contain the list of items and the list of transferred items.

        :returns: The data of the widget in JSON format: {"items": List[Dict[str, Any]], "transferred_items": List[str]}.
            "items" - the list of items in the widget in JSON format. Each item is represented as item.
            "transferred_items" - the list of transferred items (keys of the items which should be displayed in the right list).
        :rtype: Dict[str, List[Dict[str, Union[str, bool]]]]
        """

        res = {
            "items": [],
        }
        if self._items:
            res["items"] = [item.to_json() for item in self._items]

        return res

    def get_json_state(self) -> Dict[str, List[str]]:
        """
        Returns the state of the widget in JSON format.

        State will contain the list of transferred items.

        :returns: The state of the widget in JSON format: {"transferred_items": List[str]}. "transferred_items" - the list of
            transferred items (keys of the items which should be displayed in the right list).
        :rtype: Dict[str, List[str]]
        """

        transferred_items = self._transferred_items

        return {"transferred_items": transferred_items}

    def get_transferred_items(self) -> List[str]:        
        """
        Returns the list of transferred items.

        :returns: List of transferred items (keys of the items which should be displayed in the right list).
        :rtype: List[str]
        """

        return StateJson()[self.widget_id]["transferred_items"]

    def get_untransferred_items(self) -> List[str]:
        """
        Returns the list of untransferred items.

        :returns: List of untransferred items (keys of the items which should be displayed in the left list).
        :rtype: List[str]
        """

        return [item.key for item in self._items if item.key not in self.get_transferred_items()]

    def value_changed(self, func: Callable) -> Callable:
        """
        Decorates a function which will be called when the the items in right list are changed (moved in or out of the list).

        :param func: Function to be wrapped with the decorator. The function should have one argument which will contain
            namedtuple with the following fields: transferred_items, untransferred_items.
        :type func: Callable

        :returns: Wrapped function.
        :rtype: Callable

        :Usage Example:

            .. code-block:: python

                tr = Transfer(items=["item1", "item2", "item3"], transferred_items=["item1"])

                # Move "item2" from the left list to the right list.
                @tr.value_changed
                def func(items):
                    print(items.transferred_items) # ["item1", "item2"]
                    print(items.untransferred_items) # ["item3"]
        """

        route_path = self.get_route_path(Transfer.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():

            Items = namedtuple("Items", ["transferred_items", "untransferred_items"])

            res = Items(transferred_items=self.get_transferred_items(), untransferred_items=self.get_untransferred_items())

            func(res)

        return _click

    def set_items(self, items: Union[List[Transfer.Item], List[str]]):
        """
        Sets the list of items for the widget.

        If the list of items is specified as strings, they will be converted to items.
        Note: this method will REPLACE the current list of items with the new one. If you want to add new items to the current
        list, use .add() method.

        :param items: List of items to be set as the new list of items for the widget. Can be specified as items
            objects or as strings of the item keys.
        :type items: Union[List[:class:`~supervisely.app.widgets.transfer.transfer.Transfer.Item`], List[str]]

        :Usage Example:

            .. code-block:: python

                tr = Transfer(items=["cat", "dog"])
                print(tr.get_untransferred_items()) # ["cat", "dog"]

                tr.set(items=["bird", "mouse"])
                print(tr.get_untransferred_items()) # ["bird", "mouse"]

                # As you can see, the list of items was replaced with the new one.
        """

        if items:
            self._items = self.__checked_items(items)
        else:
            self._items = []

        self.update_data()
        DataJson().send_changes()

    def set_transferred_items(self, transferred_items: List[str]):
        """
        Sets the list of transferred items.
        The list should contain only the keys of the items specified in the list of items. Otherwise, an error will be raised.
        :param transferred_items: List of keys of the items which should be displayed in the right list.
        :type transferred_items: List[str]
        """

        self._transferred_items = self.__checked_transferred_items(transferred_items)
        self.update_state()
        StateJson().send_changes()

    def add(self, items: Union[List[Item], List[str]]):
        """
        Adds new items to the current list of items.

        If the list of items is specified as strings, items will be created from them. If the list of adding
        items contains any items with the same key as the items in the current list, an error will be raised.

        :param items: List of items to be added to the current list of items. Can be specified as items or
            as strings of the item keys.
        :type items: Union[List[:class:`~supervisely.app.widgets.transfer.transfer.Transfer.Item`], List[str]]

        :raises ValueError: If the list of adding items contains any items with the same key as the items in the current list.

        :Usage Example:

            .. code-block:: python

                tr = Transfer(items=["cat", "dog"])
                print(tr.get_untransferred_items()) # ["cat", "dog"]

                tr.add(items=["bird", "mouse"])
                print(tr.get_untransferred_items()) # ["cat", "dog", "bird", "mouse"]
        """

        items = self.__checked_items(items)

        if any([item.key in [item.key for item in self._items] for item in items]):
            raise ValueError("The 'items' argument should not contain any items with the same key as the items in the current list.")
        else:
            self._items.extend(items)
            self.update_data()
            DataJson().send_changes()

    def remove(self, items_keys: List[str]):
        """
        Removes items from the current list of items.

        The list of items to be removed should contain keys of the items which should be removed. If there are no items with 
        the specified keys in the current list, nothing will be removed and no error will be raised.

        :param items_keys: List of keys of the items to be removed.
        :type items_keys: List[str]
        """

        self._items = [item for item in self._items if item.key not in items_keys]
        self._transferred_items = [item for item in self._transferred_items if item not in items_keys]
        self.update_data()
        self.update_state()
        DataJson().send_changes()
        StateJson().send_changes()

    def get_items_keys(self) -> List[str]:
        """
        Returns the list of keys of the items.

        :returns: List of keys of the items.
        :rtype: List[str]
        """

        return [item.key for item in self._items]
