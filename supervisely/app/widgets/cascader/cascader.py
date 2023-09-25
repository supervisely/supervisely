from __future__ import annotations
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget
from typing import List, Dict

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
        filterable: bool = False,
        placeholder: str = "select",
        size: Literal["large", "small", "mini"] = None,
        expand_trigger: Literal["click", "hover"] = "click",
        disabled: bool = False,
        clearable: bool = False,
        show_all_levels: bool = True,
        change_on_select: bool = False,
        selected_options: List[str] = None,
        widget_id: str = None,
    ):

        self._items = items
        self._filterable = filterable
        self._placeholder = placeholder
        self._changes_handled = False
        self._size = size
        self._expand_trigger = expand_trigger
        self._disabled = disabled
        self._clearable = clearable
        self._show_all_levels = show_all_levels
        self._change_on_select = change_on_select
        self._selected_options = selected_options
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _set_items(self):
        return [item.to_json() for item in self._items]

    def get_json_data(self) -> Dict:
        res = {
            "filterable": self._filterable,
            "placeholder": self._placeholder,
            "expand_trigger": self._expand_trigger,
            "size": self._size,
            "disabled": self._disabled,
            "clearable": self._clearable,
            "show_all_levels": self._show_all_levels,
            "change_on_select": self._change_on_select,
            "props": {
                "label": "label",
                "value": "value",
                "children": "children",
                "disabled": "disabled",
            },
        }
        if self._items is not None:
            res["options"] = self._set_items()

        return res

    def get_json_state(self) -> Dict:
        return {"selectedOptions": self._selected_options}

    def get_value(self):
        return StateJson()[self.widget_id]["selectedOptions"]

    def set_value(self, value: List[str]):
        self._selected_options = value
        StateJson()[self.widget_id]["selectedOptions"] = self._selected_options
        StateJson().send_changes()

    def get_items(self):
        return DataJson()[self.widget_id]["options"]

    def set_items(self, value: List[Cascader.Item]):
        self._items = value
        DataJson()[self.widget_id]["options"] = self._set_items()
        DataJson().send_changes()

    def add_items(self, value: List[Cascader.Item]):
        self._items.extend(value)
        DataJson()[self.widget_id]["options"] = self._set_items()
        DataJson().send_changes()

    def expand_to_hover(self):
        self._expand_trigger = "hover"
        DataJson()[self.widget_id]["expand_trigger"] = self._expand_trigger
        DataJson().send_changes()

    def expand_to_click(self):
        self._expand_trigger = "click"
        DataJson()[self.widget_id]["expand_trigger"] = self._expand_trigger
        DataJson().send_changes()

    def set_disabled(self):
        self._disabled = True
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def set_unabled(self):
        self._disabled = False
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(Cascader.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            func(res)

        return _click
