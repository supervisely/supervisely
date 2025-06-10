from typing import Dict, Optional

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class RandomSplitsTable(Widget):
    def __init__(
        self,
        items_count: int,
        start_train_percent: Optional[int] = 80,
        disabled: Optional[bool] = False,
        widget_id: Optional[int] = None,
    ):
        self._disabled = disabled
        if 1 <= start_train_percent <= 99:
            pass
        else:
            raise ValueError("start_train_percent must be in range [1; 99].")
        self._table_data = [
            {"name": "train", "type": "success"},
            {"name": "val", "type": "primary"},
            {"name": "total", "type": "gray"},
        ]
        if items_count is None:
            items_count = 0
        self._items_count = items_count
        train_count = int(items_count / 100 * start_train_percent)
        self._count = {"total": items_count, "train": train_count, "val": items_count - train_count}

        self._percent = {
            "total": 100,
            "train": start_train_percent,
            "val": 100 - start_train_percent,
        }

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "table_data": self._table_data,
            "items_count": self._items_count,
            "disabled": self._disabled,
        }

    def get_json_state(self):
        return {"count": self._count, "percent": self._percent}

    def get_splits_counts(self) -> Dict[str, int]:
        return StateJson()[self.widget_id]["count"]

    def set_items_count(self, items_count: int):
        self._items_count = items_count
        current_train_percent = self._percent["train"]
        train_count = int(items_count * current_train_percent / 100)
        self._count = {"total": items_count, "train": train_count, "val": items_count - train_count}

        DataJson()[self.widget_id]["items_count"] = self._items_count
        DataJson().send_changes()
        StateJson()[self.widget_id]["count"] = self._count
        StateJson().send_changes()

    def set_train_split_percent(self, percent: int):
        if 1 <= percent <= 99:
            self._percent["train"] = percent
            self._percent["val"] = 100 - percent
            self._count["train"] = int(self._items_count / 100 * percent)
            self._count["val"] = self._items_count - self._count["train"]

            StateJson()[self.widget_id]["count"] = self._count
            StateJson()[self.widget_id]["percent"] = self._percent
            StateJson().send_changes()
        else:
            raise ValueError("percent must be in range [1; 99].")

    def get_train_split_percent(self) -> int:
        return StateJson()[self.widget_id]["percent"]["train"]

    def set_val_split_percent(self, percent: int):
        if 1 <= percent <= 99:
            self._percent["val"] = percent
            self._percent["train"] = 100 - percent
            self._count["val"] = int(self._items_count / 100 * percent)
            self._count["train"] = self._items_count - self._count["val"]
            StateJson()[self.widget_id]["count"] = self._count
            StateJson()[self.widget_id]["percent"] = self._percent
            StateJson().send_changes()
        else:
            raise ValueError("percent must be in range [1; 99].")

    def get_val_split_percent(self) -> int:
        return StateJson()[self.widget_id]["percent"]["val"]
