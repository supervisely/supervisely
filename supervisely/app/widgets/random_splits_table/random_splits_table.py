from typing import Dict, Optional

from supervisely.app import StateJson
from supervisely.app.widgets import Widget

class RandomSplitsTable(Widget):
    def __init__(self, items_count: int, widget_id: Optional[int] = None):
        self._table_data = [
            {"name": "train", "type": "success"},
            {"name": "val", "type": "primary"},
            {"name": "total", "type": "gray"},
        ]
        self._items_count = items_count
        train_percent = 80
        train_count = int(items_count / 100 * train_percent)
        self._count = {
            "total": items_count,
            "train": train_count,
            "val": items_count - train_count
        }

        self._percent = {
            "total": 100,
            "train": train_percent,
            "val": 100 - train_percent
        }
        self._disabled = False
        self._share_items = False

        super().__init__(widget_id=widget_id, file_path=__file__)


    def get_json_data(self) -> Dict:
        return {
            "table_data": self._table_data,
            "items_count": self._items_count,
        }

    def get_json_state(self) -> Dict:
        return {
            "count": self._count,
            "percent": self._percent,
            "disabled": self._disabled,
            "shareImagesBetweenSplits": self._share_items
        }