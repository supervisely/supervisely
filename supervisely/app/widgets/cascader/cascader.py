from __future__ import annotations
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget, ConditionalWidget
from typing import List, Dict, Optional

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
        size: Literal["large", "small", "mini"] = "small",
        hover: Literal["click", "hover"] = "click",
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
        self._hover = hover
        self._disabled = disabled
        self._clearable = clearable
        self._show_all_levels = show_all_levels
        self._change_on_select = change_on_select
        self._selected_options = selected_options
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {
            "filterable": self._filterable,
            "placeholder": self._placeholder,
            "hover": self._hover,
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
            res["options"] = [item.to_json() for item in self._items]

        return res

    def get_json_state(self) -> Dict:
        return {"selectedOptions": self._selected_options}

    def get_value(self):
        return StateJson()[self.widget_id]["selectedOptions"]

    def value_changed(self, func):
        route_path = self.get_route_path(Cascader.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            func(res)

        return _click
