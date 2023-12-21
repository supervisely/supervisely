from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Union

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class Collapse(Widget):
    """A Collapse widget allows users to efficiently manage and navigate content by toggling between
    hidden and visible states, promoting a more compact and organized user interface.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/layouts-and-containers/collapse>`_
        (including screenshots and examples).

    :param items: list of items to be displayed in the widget
    :type items: List[Collapse.Item]
    :param accordion: if True, only one panel could be active at a time
    :type accordion: bool
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
        .. code-block:: python

        from supervisely.app.widgets import Collapse

        items = [
            Collapse.Item("item1", "Item 1", "Content of item 1"),
            Collapse.Item("item2", "Item 2", "Content of item 2"),
        ]
        collapse = Collapse(items)

    """

    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item(object):
        """Represents an item of Collapse widget.

        :param name: unique identification of the panel
        :type name: str
        :param title: title of the panel
        :type title: str
        :param content: content of the panel
        :type content: Optional[Union[Widget, str]]
        """

        def __init__(self, name: str, title: str, content: Optional[Union[Widget, str]]):
            self.name = name
            self.title = title
            self.content = content

        def to_json(self) -> Dict[str, Any]:
            """Returns JSON representation of the item.
            Dictionary contains the following keys:
                - name: unique identification of the panel
                - label: title of the panel
                - content_type: type of the content (str or Widget)

            :return: JSON representation of the item
            :rtype: Dict[str, Any]
            """
            if isinstance(self.content, str):
                content_type = "text"
            else:
                content_type = str(type(self.content))
            return {
                "name": self.name,
                "label": self.title,
                "content_type": content_type,
            }

    def __init__(
        self,
        items: Optional[List[Collapse.Item]] = None,
        accordion: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        if items is None:
            items = [Collapse.Item("default", "Empty default item", "")]

        labels = [item.name for item in items]
        if len(set(labels)) != len(labels):
            raise ValueError("All items must have a unique name.")

        self._items: List[Collapse.Item] = items

        self._accordion = accordion
        self._active_panels = []

        self._items_names = set(labels)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_items_json(self) -> List[Dict[str, Any]]:
        return [item.to_json() for item in self._items]

    def get_json_data(self) -> Dict[str, Any]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - accordion: if True, only one panel could be active at a time
            - items: list of items to be displayed in the widget

        :return: dictionary with widget data
        :rtype: Dict[str, Any]
        """
        return {
            "accordion": self._accordion,
            "items": self._get_items_json(),
        }

    def get_json_state(self) -> Dict[str, List[str]]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - value: list of active panels

        :return: dictionary with widget state
        :rtype: Dict[str, List[str]]
        """
        return {"value": self._active_panels}

    def set_active_panel(self, value: Union[str, List[str]]) -> None:
        """Set active panel or panels.

        :param value: panel name(s)
        :type value: Union[str, List[str]]
        :raises TypeError: value of type List[str] can't be setted, if accordion is True.
        :raises ValueError: panel with such title doesn't exist.
        """
        if isinstance(value, list):
            if self._accordion:
                raise TypeError(
                    "Only one panel could be active in accordion mode. Use `str`, not `list`."
                )
            for name in value:
                if name not in self._items_names:
                    raise ValueError(
                        f"Can't activate panel `{name}`: item with such name doesn't exist."
                    )
        else:
            if value not in self._items_names:
                raise ValueError(
                    f"Can't activate panel `{value}`: item with such name doesn't exist."
                )

        if isinstance(value, str):
            self._active_panels = [value]
        else:
            self._active_panels = value

        StateJson()[self.widget_id]["value"] = self._active_panels
        StateJson().send_changes()

    def get_active_panel(self) -> Union[str, List[str]]:
        """Returns active panel or panels.

        :return: active panel or panels
        :rtype: Union[str, List[str]]
        """
        return StateJson()[self.widget_id]["value"]

    def get_items(self) -> List[Collapse.Item]:
        """Returns list of items.

        :return: list of items
        :rtype: List[Collapse.Item]
        """
        return DataJson()[self.widget_id]["items"]

    def set_items(self, value: List[Collapse.Item]) -> None:
        """Set items for the widget.
        This method replaces all existing items with new ones.
        To add new items, use :meth:`add_items` method.

        :param value: list of items to be displayed in the widget
        :type value: List[Collapse.Item]
        """
        names = [val.name for val in value]

        self._items_names = self._make_set_from_unique(names)
        self._items = value
        self._active_panels = []

        DataJson()[self.widget_id]["items"] = self._get_items_json()
        DataJson().send_changes()

    def add_items(self, value: List[Collapse.Item]) -> None:
        """Add items to the widget.
        This method adds new items to the existing ones.
        To replace all existing items with new ones, use :meth:`set_items` method.

        :param value: list of items to be displayed in the widget
        :type value: List[Collapse.Item]
        :raises ValueError: item with such name already exists.
        """
        names = [val.name for val in value]
        set_of_names = self._make_set_from_unique(names)

        for name in names:
            if name in self._items_names:
                raise ValueError(f"Item with name {name} already exists.")

        self._items.extend(value)
        self._items_names.update(set_of_names)
        DataJson()[self.widget_id]["items"] = self._get_items_json()
        DataJson().send_changes()

    def value_changed(self, func: Callable[[List[str]], Any]) -> Callable[[], None]:
        """Decorator for the function to be called when active panel or panels are changed.

        :param func: function to be called when active panel or panels are changed
        :type func: Callable[[List[str]], Any]
        :return: decorated function
        :rtype: Callable[[], None]
        """
        route_path = self.get_route_path(Collapse.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            active = self.get_active_panel()
            self._active_panels = active
            func(active)

        return _click

    @property
    def items_names(self) -> Set[str]:
        """Returns set of names of items.

        :return: set of names of items
        :rtype: Set[str]
        """

        return self._items_names

    def _make_set_from_unique(self, names: List[str]) -> Set[str]:
        set_of_names = set(names)
        if len(names) != len(set_of_names):
            raise ValueError("All items must have a unique name.")
        return set_of_names
