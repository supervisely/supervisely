from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class RadioCard(Widget):

    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item:
        def __init__(
            self,
            title: Optional[str] = None,
            content: Optional[Widget] = None,
            description: Optional[str] = None,
            description_content: Optional[Widget] = None,
            img: Optional[str] = None,
            tag: Optional[str] = None,
            tag_icon: Optional[str] = None,  # zmdi
            disabled: bool = False,
            disabled_text: Optional[str] = None,
            bg_color: str = "#3ab63a",
        ):
            self.title = title
            self.content = content
            self.description = description
            self.description_content = description_content
            self.img = img
            self.tag = tag
            self.tag_icon = tag_icon
            self.disabled = disabled
            self.disabled_text = disabled_text
            self.bg_color = bg_color
            self.idx = None

        def to_json(self) -> Dict[str, Any]:
            return {
                "title": self.title,
                "description": self.description,
                "img": self.img,
                "tag": self.tag,
                "tagIcon": self.tag_icon,
                "disabled": self.disabled,
                "disabledText": self.disabled_text,
                "bgColor": self.bg_color,
            }

    def __init__(
        self,
        items: Optional[List[Item]] = None,
        active_item_idx: int = 0,
        items_width: int = 190,
        widget_id: Optional[str] = None,
    ):
        self._active_item_idx = active_item_idx
        self._items_width = items_width
        self._changes_handled = False
        self._items = items or []
        for idx, item in enumerate(self._items):
            item.idx = idx
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_state(self) -> Dict[str, Any]:
        return {"activeToolIdx": 0}

    def get_json_data(self) -> Dict[str, Any]:
        return {
            "items": [item.to_json() for item in self._items],
            "items_width": self._items_width,
        }

    def get_active_idx(self) -> int:
        self._active_item_idx = StateJson()[self.widget_id]["activeToolIdx"]
        return self._active_item_idx

    def value_changed(self, func: Callable[[List[str]], Any]) -> Callable[[], None]:
        route_path = self.get_route_path(RadioCard.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            self.get_active_idx()
            func(self._active_item_idx)

        return _click
