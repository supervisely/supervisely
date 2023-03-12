from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class MessageBox(Widget):
    class Routes:
        OPEN_CLICKED = "open_cb"
        CLOSE_CLICKED = "close_cb"

    def __init__(
        self,
        title: str = "",
        type: Literal["info", "warning", "error"] = "info",
        message: str = "",
        widget_id: str = None,
    ):
        self._title = title
        self._type = type
        self._message = message
        self._open_clicked_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {"example1": {"title": self._title, "type": self._type, "message": self._message}}

    def get_json_state(self) -> Dict:
        return {"selected": []}

    def open_clicked(self, func):
        route_path = self.get_route_path(MessageBox.Routes.OPEN_CLICKED)
        server = self._sly_app.get_server()
        self._open_clicked_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_current_frame()
            func(res)

        return _click

    # def set_value(self, value: str):
    #     self._md = value
    #     DataJson()[self.widget_id]["md"] = value
    #     DataJson().send_changes()

    # def get_value(self):
    #     return DataJson()[self.widget_id]["md"]

    # def get_height(self):
    #     return DataJson()[self.widget_id]["options"]["height"]
