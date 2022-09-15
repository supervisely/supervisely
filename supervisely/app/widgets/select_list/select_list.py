from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class SelectList(Widget):
    def __init__(
            self,
            items: List[dict],
            widget_id: str = None
    ):
        self._items = items
        self._selected = None
        self._widget_id = widget_id

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return None

    def get_json_state(self):
        return {
            "items": self._items,
            "selected": self._selected,
        }

    # def set_items(self, value: List[dict]):
    #     StateJson()[self.widget_id]["items"] = value
    #     StateJson().send_changes()
    #
    # def get_items(self):
    #     return StateJson()[self.widget_id]["items"]
    #
    # def set_selected_item(self, id: int, value: str):
    #     StateJson()[self.widget_id]["selected"] = StateJson()[self.widget_id]["items"][id][value]
    #     StateJson().send_changes()
    #
    # def get_selected_item(self):
    #     return StateJson()[self.widget_id]["selected"]
