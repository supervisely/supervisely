from __future__ import annotations

from typing import Callable, Dict, List, Union

from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget


class TreeSelect(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item:
        def __init__(
            self, id: str, label: str | None = None, children: List[TreeSelect.Item] = None
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

        def from_json(data: Dict[str, Union[str, List[Dict]]]) -> TreeSelect.Item:
            return TreeSelect.Item(
                id=data["id"],
                label=data.get("label"),
                children=[TreeSelect.Item.from_json(child) for child in data.get("children", [])],
            )

        def __repr__(self):
            return f"Item(id={self.id}, label={self.label}, children={self.children})"

    def __init__(
        self,
        items: List[TreeSelect.Item] | None = None,
        multiple_select: bool = False,
        flat: bool = False,
        always_open: bool = False,
        widget_id: str | None = None,
    ):
        self._items = items or []
        self._multiple = multiple_select
        self._flat = flat
        self._always_open = always_open
        self._value_format = "object"
        self._value = [] if multiple_select else None

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "items": [item.to_json() for item in self._items],
        }

    def get_json_state(self):
        return {
            "value": self.value,
            "options": {
                "multiple": self._multiple,
                "flat": self._flat,
                "alwaysOpen": self._always_open,
                "valueFormat": self._value_format,
            },
        }

    @property
    def value(self) -> Union[List[TreeSelect.Item], TreeSelect.Item]:
        return self._value

    @value.setter
    def value(self, value: Union[List[TreeSelect.Item], TreeSelect.Item]):
        self._value = value

    def _get_value(self) -> Union[List[TreeSelect.Item], TreeSelect.Item]:
        res = StateJson()[self.widget_id]["value"]
        if isinstance(res, list):
            return [TreeSelect.Item.from_json(item) for item in res]
        return TreeSelect.Item.from_json(res)

    def _set_value(self, value: Union[List[TreeSelect.Item], TreeSelect.Item]):
        self.value = value
        StateJson()[self.widget_id]["value"] = value
        StateJson().send_changes()

    def get_selected_items(self) -> List[TreeSelect.Item]:
        if not self._multiple:
            raise ValueError("Multiple items mode is disabled, use get_selected_item instead")
        return self._get_value()

    def get_selected_item(self) -> TreeSelect.Item:
        if self._multiple:
            raise ValueError("Multiple items mode is enabled, use get_selected_items instead")
        return self._get_value()

    def set_selected_items(self, items: List[TreeSelect.Item]):
        if not self._multiple:
            raise ValueError("Multiple items mode is disabled, use set_selected_item instead")
        self._set_value(items)

    def set_selected_item(self, item: TreeSelect.Item):
        if self._multiple:
            raise ValueError("Multiple items mode is enabled, use set_selected_items instead")
        self._set_value(item)

    def _update_items(self, items: Union[List[TreeSelect.Item], TreeSelect.Item], overwrite: bool):
        if overwrite:
            self._items = items
        else:
            self._items.extend(items)
        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()

    def set_items(self, items: List[TreeSelect.Item]):
        if not self._multiple:
            raise ValueError("Multiple items mode is disabled, use set_item instead")
        self._update_items(items, overwrite=True)

    def add_items(self, items: List[TreeSelect.Item]):
        if not self._multiple:
            raise ValueError("Multiple items mode is disabled, use add_item instead")
        self._update_items(items, overwrite=False)

    def set_item(self, item: TreeSelect.Item):
        if self._multiple:
            raise ValueError("Multiple items mode is enabled, use set_items instead")
        self._update_items(item, overwrite=True)

    def clear_items(self):
        items = [] if self._multiple else None
        self._update_items(items, overwrite=True)

    def value_changed(self, func: Callable) -> Callable:
        route_path = self.get_route_path(TreeSelect.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self._get_value()

            func(res)

        return _click
