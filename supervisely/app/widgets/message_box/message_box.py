from typing import Dict

from supervisely.app import DataJson, StateJson
from supervisely.app.fastapi.utils import run_sync
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.app.widgets import Widget
from supervisely.app.widgets_context import JinjaWidgets

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class MessageBox(Widget):
    def __init__(
        self,
        title: str = "",
        message: str = "",
        type: Literal["info", "warning", "error"] = "info",
        widget_id: str = None,
    ):
        self._title = title
        self._message = message
        self._type = type

        super().__init__(widget_id=widget_id, file_path=__file__)

        script_path = "./sly/css/app/widgets/message_box/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

    def get_json_data(self) -> Dict:
        return {
            "data": {
                "title": self._title,
                "type": self._type,
                "message": self._message,
            }
        }

    def get_json_state(self) -> Dict:
        return {}

    def open(
        self,
        title: str = None,
        message: str = None,
        type: Literal["info", "warning", "error"] = None,
    ):
        data = DataJson()[self.widget_id]["data"]
        if title is not None:
            data["title"] = title
        if message is not None:
            data["message"] = message
        if type is not None:
            if type not in ["info", "warning", "error"]:
                raise ValueError("type should be one of ['info', 'warning', 'error']")
            data["type"] = type
        DataJson().send_changes()
        StateJson().send_changes()

        run_sync(
            WebsocketManager().broadcast(
                {
                    "runAction": {
                        "action": f"message-box-{self.widget_id}",
                        "payload": {"data": data},
                    }
                }
            )
        )

    def set_title(self, title: str):
        self._set({"title": title})

    def get_title(self):
        return DataJson()[self.widget_id]["data"]["title"]

    def set_message(self, message: str):
        self._set({"message": message})

    def get_message(self):
        return DataJson()[self.widget_id]["data"]["message"]

    def set_type(self, value: Literal["info", "warning", "error"]):
        if value not in ["info", "warning", "error"]:
            raise ValueError("Value should be one of ['info', 'warning', 'error']")
        self._set({"type": value})

    def get_type(self):
        return DataJson()[self.widget_id]["data"]["type"]

    def set(self, data: dict):
        if type(data) is not dict:
            raise TypeError("data should be dict")
        new_type = data.get("type")
        if new_type is not None and new_type not in ["info", "warning", "error"]:
            raise ValueError("type should be one of ['info', 'warning', 'error']")
        self._set(data)

    def get_data(self):
        return DataJson()[self.widget_id]["data"]

    def _set(self, data):
        if data.get("title") is None:
            data["title"] = self._title
        else:
            self._title = data.get("title")
        if data.get("message") is None:
            data["message"] = self._message
        else:
            self._message = data.get("message")
        if data.get("type") is None:
            data["type"] = self._type
        else:
            self._type = data.get("type")

        DataJson()[self.widget_id]["data"] = data
        DataJson().send_changes()
