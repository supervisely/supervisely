from functools import wraps

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app.widgets import Widget


class MyText(Widget):
    def __init__(
        self,
        text: str = "My textarea",
        widget_id=None,
    ):
        self._text = text
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "text": self._text,
        }

    def get_json_state(self):
        return None


