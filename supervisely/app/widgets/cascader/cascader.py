from __future__ import annotations
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget
from typing import List, Dict, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Cascader(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item:
        def __init__(
            self,
            value,
            label: str = None,
            children: List[Cascader.Item] = [],
            disabled: bool = False,
        ) -> Cascader.Item:
            self.value = value
            self.label = label
            if label is None:
                self.label = str(self.value)
            self.children = children
            self.disabled = disabled

        def to_json(self):
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
        items: List[Cascader.Item] = None,
        selected_items: List[str] = None,
        filterable: bool = False,
        placeholder: str = "select",
        size: Literal["large", "small", "mini"] = None,
        expand_trigger: Literal["click", "hover"] = "click",
        clearable: bool = True,
        show_all_levels: bool = True,
        parent_selectable: bool = False,
        widget_id: str = None,
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

    def get_json_data(self) -> Dict:
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

    def get_json_state(self) -> Dict:
        return {"selectedItems": self._selected_items}

    def get_selected_items(self):
        return StateJson()[self.widget_id]["selectedItems"]

    def select_item(self, values: List[Union[str, Cascader.Item]]):
        """
        values example:
        values = ["cat", "black cat", "fluffy cat"]
        or
        values = [Cascader.Item("cat"), Cascader.Item("black cat"), Cascader.Item("fluffy cat")]
        """

        str_values = []
        for item in values:
            if type(item) == Cascader.Item:
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

    def deselect(self):
        self.select_item([])

    def get_items(self):
        return DataJson()[self.widget_id]["items"]

    def get_item(self, value: str):
        def _recursive_search(items):
            for item in items:
                if item.value == value:
                    return item
                found = _recursive_search(item.children)
                if found is not None:
                    return found
            return None

        return _recursive_search(self._items)

    def set_items(self, items: List[Cascader.Item]):
        if any(type(item) != Cascader.Item for item in items):
            raise TypeError("All items must be of type Cascader.Item")
        self._items = items
        self._set_items()
        self.deselect()

    def add_item(self, item: Cascader.Item):
        self.add_items([item])

    def add_items(self, items: List[Cascader.Item]):
        if any(type(item) != Cascader.Item for item in items):
            raise TypeError("All items must be of type Cascader.Item")
        for item in items:
            if item.value in [i.value for i in self._items]:
                raise ValueError(f"Item with value '{item.value}' already exists.")
        self._items.extend(items)
        self._set_items()

    def value_changed(self, func):
        route_path = self.get_route_path(Cascader.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_selected_items()
            func(res)

        return _click
