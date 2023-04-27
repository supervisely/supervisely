from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import Dict


class Markdown(Widget):
    def __init__(self, content: str = "", height: int = 300, widget_id: str = None):
        self._md = content
        self._height = f"{height}px"

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {"md": self._md, "options": {"height": self._height}}

    def get_json_state(self) -> Dict:
        return {}

    def set_value(self, value: str):
        self._md = value
        DataJson()[self.widget_id]["md"] = value
        DataJson().send_changes()

    def get_height(self):
        return DataJson()[self.widget_id]["options"]["height"]
