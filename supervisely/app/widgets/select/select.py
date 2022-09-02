from __future__ import annotations
from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from typing import List, Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Select(Widget):
    class Item:
        def __init__(self, value, label: str = None) -> None:
            self.value = value
            self.label = label
            if label is None:
                self.label = str(self.value)

        def to_json(self) -> Dict:
            return {"label": self.label, "value": self.value}

    class Group:
        def __init__(self, label, items: List[Select.Item] = None) -> None:
            self.label = label
            self.items = items

        def to_json(self) -> Dict:
            return {
                "label": self.label,
                "options": [item.to_json() for item in self.items],
            }

    def __init__(
        self,
        items: List[Select.Item] = None,
        groups: List[Select.Group] = None,
        filterable: bool = False,
        placeholder: str = "select",
        widget_id: str = None,
    ):
        if items is None and groups is None:
            raise ValueError("One of the arguments has to be defined: items or groups")

        if items is not None and groups is not None:
            raise ValueError(
                "Only one of the arguments has to be defined: items or groups"
            )

        self._items = items
        self._groups = groups
        self._filterable = filterable
        self._placeholder = placeholder

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_first_value(self) -> Select.Item:
        if self._items is not None:
            return self._items[0]
        if self._groups is not None:
            return self._groups[0][0]

    def get_json_data(self):
        res = {
            "filterable": self._filterable,
            "placeholder": self._placeholder,
        }
        if self._items is not None:
            res["items"] = [item.to_json() for item in self._items]
        if self._groups is not None:
            res["groups"] = [group.to_json() for group in self._groups]
        return res

    def get_json_state(self):
        return {"value": self._get_first_value().value}
