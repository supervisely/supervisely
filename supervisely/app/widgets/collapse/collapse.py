from __future__ import annotations
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Union, Dict, Any, Optional


class Collapse(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item(object):
        def __init__(self, content: Union[Widget, str], title: str = "") -> None:
            self.name = title  # unique identification of the panel
            self.title = title
            self.content = content

        def to_json(self) -> Dict[str, Any]:
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
        labels: List[str],
        contents: List[Union[str, Widget]],
        accordion: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        if len(labels) != len(contents):
            raise ValueError("labels length must be equal to contents length in Collapse widget.")
        if len(set(labels)) != len(labels):
            raise ValueError("All of collapse labels should be unique.")

        self._items: List[Collapse.Item] = []
        for title, content in zip(labels, contents):
            self._items.append(Collapse.Item(title=title, content=content))

        self._accordion = accordion
        if self._accordion:
            self._active_panels = labels[0]
        else:
            self._active_panels = [labels[0]]

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_items_json(self) -> List[Dict[str, Any]]:
        return [item.to_json() for item in self._items]

    def get_json_data(self):
        return {
            "accordion": self._accordion,
            "items": self._get_items_json(),
        }

    def get_json_state(self):
        return {"value": self._active_panels}

    def set_active_panel(self, value: Union[str, List[str]]):
        """Set active panel.

        Args:
            value (Union[str, List[str]]): str if accordion mode,
                else List[str]

        Raises:
            TypeError: value of type List[str] can't be setted, if accordion is True.
        """
        if isinstance(value, list) and self._accordion:
            raise TypeError(
                "Only one panel could be active in accordion mode. Use `str`, not `list`."
            )

        if isinstance(value, str) and not self._accordion:
            self._active_panels = [value]
        else:
            self._active_panels = value

        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_active_panel(self) -> Union[str, List[str]]:
        return StateJson()[self.widget_id]["value"]

    def get_items(self):
        return DataJson()[self.widget_id]["items"]

    def set_items(self, value: List[Collapse.Item]):
        self._items = value
        DataJson()[self.widget_id]["items"] = self._set_items()
        DataJson().send_changes()

    def add_items(self, value: List[Collapse.Item]):
        self._items.extend(value)
        DataJson()[self.widget_id]["items"] = self._set_items()
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(Collapse.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_active_panel()
            self._value = res
            func(res)

        return _click
