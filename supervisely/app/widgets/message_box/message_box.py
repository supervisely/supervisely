from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class MessageBox(Widget):
    def __init__(
        self,
        title: str = "Message Box",
        message: str = "",
        type: Literal["info", "warning", "error", "text"] = "text",
        button_text: str = "Open MessageBox",
        widget_id: str = None,
    ):
        self._title = title
        self._message = message
        self._button_text = button_text
        self._type = type

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {
            "messageBox": {"title": self._title, "type": self._type, "message": self._message},
        }

    def get_json_state(self) -> Dict:
        return {}

    def set_message(self, text: str, type: Literal["info", "warning", "error", "text"] = "text"):
        self._message = text
        self._type = type
        DataJson()[self.widget_id]["messageBox"]["message"] = text
        DataJson()[self.widget_id]["messageBox"]["type"] = type
        DataJson().send_changes()

    def get_message(self):
        return DataJson()[self.widget_id]["messageBox"]["message"]

    def get_type(self):
        return DataJson()[self.widget_id]["messageBox"]["type"]
