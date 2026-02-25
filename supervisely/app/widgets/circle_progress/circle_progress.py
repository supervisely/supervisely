from typing import Literal
from supervisely.app.widgets import Widget, Progress
from supervisely.app.content import DataJson


class CircleProgress(Widget):
    """Circular progress indicator widget showing Progress status (success/exception/none)."""

    def __init__(self, progress: Progress, widget_id=None):
        """
        :param progress: Progress object to display status for.
        :type progress: :class:`~supervisely.app.widgets.progress.Progress`
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional
        """
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
