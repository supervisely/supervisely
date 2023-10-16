from typing import Any
from supervisely.app.widgets import Widget


class Draggable(Widget):
    def __init__(self, content: Widget, key: Any = None, widget_id: str = None):
        self._content = content
        self._key = key
        if key is None:
            self._key = content.widget_id
        super().__init__(widget_id=widget_id, file_path=__file__)
    
    def get_json_data(self):
        return {}
    
    def get_json_state(self):
        return {}
