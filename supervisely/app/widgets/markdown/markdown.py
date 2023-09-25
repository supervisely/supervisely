from typing import Optional

from supervisely.app.content import DataJson
from supervisely.app.widgets import Widget


class Markdown(Widget):
    """A widget that renders Markdown text."""

    def __init__(self, md: str, height: Optional[str] = None, widget_id=None):
        self.md = md
        self._height = height
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        options = {}
        if self._height:
            options["height"] = self._height
        return {
            "md": self.md,
            "options": options,
        }
    
    def get_json_state(self):
        return {}
    
    def set(self, md: str):
        self.md = md
        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()

    def set_height(self, height: str):
        self._height = height
        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()