from __future__ import annotations
from supervisely.app import StateJson
from supervisely.app.widgets import Widget
from typing import List, Dict

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


class Select(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item:
        def __init__(self, value, label: str = None) -> Select.Item:
            self.value = value
            self.label = label
            if label is None:
                self.label = str(self.value)

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
        widget_id: str = None,
    ) -> Select:
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
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_first_value(self) -> Select.Item:
        if self._items is not None:
            return self._items[0]
        if self._groups is not None:
            return self._groups[0].items[0]

    def get_json_data(self) -> Dict:
        res = {
            "filterable": self._filterable,
            "placeholder": self._placeholder,
            "items": None,
            "groups": None,
        }
        if self._items is not None:
            res["items"] = [item.to_json() for item in self._items]
        if self._groups is not None:
            res["groups"] = [group.to_json() for group in self._groups]
        return res

    def get_json_state(self) -> Dict:
        return {"value": self._get_first_value().value}

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
