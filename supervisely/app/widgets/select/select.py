from __future__ import annotations
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget, ConditionalWidget
from typing import List, Dict, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# r = sly.app.widgets.Text(text="right part", status="error")
# items = [
#     sly.app.widgets.Select.Item(value="option1"),
#     sly.app.widgets.Select.Item(value="option2"),
#     sly.app.widgets.Select.Item(value="option3"),
# ]
# r = sly.app.widgets.Select(items=items, filterable=True, placeholder="select me")


# groups = [
#     sly.app.widgets.Select.Group(
#         label="group1",
#         items=[
#             sly.app.widgets.Select.Item(value="g1-option1"),
#             sly.app.widgets.Select.Item(value="g1-option2"),
#         ],
#     ),
#     sly.app.widgets.Select.Group(
#         label="group2",
#         items=[
#             sly.app.widgets.Select.Item(value="g2-option1"),
#             sly.app.widgets.Select.Item(value="g2-option2"),
#         ],
#     ),
# ]
# r = sly.app.widgets.Select(groups=groups, filterable=True, placeholder="select me")


# @r.value_changed
# def do(value):
#     print(f"new value is: {value}")


class Select(ConditionalWidget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item:
        def __init__(self, value, label: str = None, content: Widget = None) -> Select.Item:
            self.value = value
            self.label = label
            if label is None:
                self.label = str(self.value)
            self.content = content

        def to_json(self):
            return {"label": self.label, "value": self.value}

    class Group:
        def __init__(self, label, items: List[Select.Item] = None) -> Select.Item:
            self.label = label
            self.items = items

        def to_json(self):
            res = {
                "label": self.label,
                "options": [item.to_json() for item in self.items],
            }
            return res

    def __init__(
        self,
        items: List[Select.Item] = None,
        groups: List[Select.Group] = None,
        filterable: bool = False,
        placeholder: str = "select",
        size: Literal["large", "small", "mini"] = None,
        multiple: bool = False,
        widget_id: str = None,
    ) -> Select:
        if items is None and groups is None:
            raise ValueError("One of the arguments has to be defined: items or groups")

        if items is not None and groups is not None:
            raise ValueError("Only one of the arguments has to be defined: items or groups")

        self._groups = groups
        self._filterable = filterable
        self._placeholder = placeholder
        self._changes_handled = False
        self._size = size
        self._multiple = multiple

        super().__init__(items=items, widget_id=widget_id, file_path=__file__)

    def _get_first_value(self) -> Select.Item:
        if self._items is not None and len(self._items) > 0:
            return self._items[0]
        if self._groups is not None and len(self._groups) > 0 and len(self._groups[0].items) > 0:
            return self._groups[0].items[0]
        return None

    def get_json_data(self) -> Dict:
        res = {
            "filterable": self._filterable,
            "placeholder": self._placeholder,
            "multiple": self._multiple,
            "items": None,
            "groups": None,
        }
        if self._items is not None:
            res["items"] = [item.to_json() for item in self._items]
        if self._groups is not None:
            res["groups"] = [group.to_json() for group in self._groups]
        if self._size is not None:
            res["size"] = self._size
        return res

    def get_json_state(self) -> Dict:
        first_item = self._get_first_value()
        value = None
        if first_item is not None:
            value = first_item.value
        return {"value": value}

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

    def value_changed(self, func):
        route_path = self.get_route_path(Select.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            func(res)

        return _click

    def get_items(self) -> List[Select.Item]:
        res = []
        if self._items is not None:
            res.extend(self._items)
        if self._groups is not None:
            for group in self._groups:
                res.extend(group.items)
        return res

    def set(self, items: List[Select.Item] = None, groups: List[Select.Group] = None):
        self._items = items
        self._groups = groups
        self.update_data()
        self.update_state()
        DataJson().send_changes()
        StateJson().send_changes()


class SelectString(Select):
    def __init__(
        self, 
        values: List[str], 
        labels: Optional[List[str]] = None,
        filterable: Optional[bool] = False,
        placeholder: Optional[str] = "select",
        size: Optional[Literal["large", "small", "mini"]] = None,
        multiple: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        if labels is not None:
            if len(values) != len(labels):
                raise ValueError("values length must be equal to labels length.")
            items = []
            for value, label in zip(values, labels):
                items.append(Select.Item(value, label))
        else:
            items = [Select.Item(value) for value in values]

        super(SelectString, self).__init__(
            items=items,
            groups=None,
            filterable=filterable,
            placeholder=placeholder,
            multiple=multiple,
            size=size,
            widget_id=widget_id
        )

    def _get_first_value(self) -> Select.Item:
        if self._items is not None and len(self._items) > 0:
            return self._items[0]
        return None

    def get_items(self) -> List[str]:
        return [item.value for item in self._items]

    def set(self, values: List[str], labels: Optional[List[Select.Item]] = None):
        if labels is not None:
            if len(values) != len(labels):
                raise ValueError("values length must be equal to labels length.")
            self._items = []
            for value, label in zip(values, labels):
                self._items.append(Select.Item(value, label))
        else:
            self._items = [Select.Item(value) for value in values]
        self.update_data()
        self.update_state()
        DataJson().send_changes()
        StateJson().send_changes()