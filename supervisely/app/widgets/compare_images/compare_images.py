from supervisely.annotation.annotation import Annotation
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class CompareImages(Widget):
    def __init__(
        self,
        widget_left: Widget = "Left image is not selected",
        widget_right: Widget = "Right image is not selected",
        widget_id: str = None,
    ):
        self._left = widget_left
        self._right = widget_right

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}
