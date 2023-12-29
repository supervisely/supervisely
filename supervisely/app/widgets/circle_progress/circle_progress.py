from typing import Literal
from supervisely.app.widgets import Widget, Progress
from supervisely.app.content import DataJson


class CircleProgress(Widget):
    def __init__(self, progress: Progress, widget_id=None):
        self.progress = progress
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "status": None,
        }

    def get_json_state(self):
        return {}

    def set_status(self, status: Literal["success", "exception", "none"]):
        if status == "none":
            status = None
        DataJson()[self.widget_id]["status"] = status
        DataJson().send_changes()
