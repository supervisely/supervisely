from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import Dict


class Docstring(Widget):
    def __init__(self, data: str = "", widget_id: str = None):
        self._data = data

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {"content": self._data}

    def get_json_state(self) -> Dict:
        return {}

    def set_value(self, value: str):
        self._data = value
        DataJson()[self.widget_id]["content"] = value
        DataJson().send_changes()

    def get_value(self):
        return DataJson()[self.widget_id]["content"]
