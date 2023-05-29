from __future__ import annotations
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Union, Dict, Any, Optional


class Collapse(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item(object):
        def __init__(self, name: str, title: str, content: Union[Widget, str]) -> None:
            self.name = name  # unique identification of the panel
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
        items: List[Collapse.Item],
        accordion: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        labels = [item.name for item in items]
        if len(set(labels)) != len(labels):
            raise ValueError("All of collapse names should be unique.")

        self._items: List[Collapse.Item] = items

        self._accordion = accordion
        if self._accordion:
            self._active_panels = labels[0]
        else:
            self._active_panels = [labels[0]]

        self._items_title = set(labels)
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
        """Set active panel or panels.

        :param value: panel name(s);
        :type value: Union[str, List[str]]
        :raises TypeError: value of type List[str] can't be setted, if accordion is True.
        :raises ValueError: panel with such title doesn't exist.
        """
        if isinstance(value, list):
            if self._accordion:
                raise TypeError(
                    "Only one panel could be active in accordion mode. Use `str`, not `list`."
                )
            for title in value:
                if title not in self._items_title:
                    raise ValueError(
                        f"Can't activate panel `{title}`: item with such title doesn't exist."
                    )
        else:
            if value not in self._items_title:
                raise ValueError(
                    f"Can't activate panel `{value}`: item with such title doesn't exist."
                )

        if isinstance(value, str) and not self._accordion:
            self._active_panels = [value]
        else:
            self._active_panels = value

        StateJson()[self.widget_id]["value"] = self._active_panels
        StateJson().send_changes()

    def get_active_panel(self) -> Union[str, List[str]]:
        return StateJson()[self.widget_id]["value"]

    def get_items(self):
        return DataJson()[self.widget_id]["items"]

    # def set_items(self, value: List[Collapse.Item]):
    #     self._items = value
    #     self._items_title = set([val.title for val in value])
    #     DataJson()[self.widget_id]["items"] = self._get_items_json()
    #     DataJson().send_changes()

    # def add_items(self, value: List[Collapse.Item]):
    #     self._items.extend(value)
    #     titles = [val.title for val in value]
    #     self._items_title.update(titles)
    #     DataJson()[self.widget_id]["items"] = self._get_items_json()
    #     DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(Collapse.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            active = self.get_active_panel()
            self._active_panels = active
            func(active)

        return _click
