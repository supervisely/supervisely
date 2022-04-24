import copy
import functools
from enum import Enum
from typing import Literal

import fastapi
from varname import varname

from supervisely.app.widgets import Widget


class ElementButton(Widget):
    class Routes:
        BUTTON_CLICKED = 'button_clicked_cb'

    def __init__(self,
                 text: str = 'Element Button',
                 button_type: Literal["primary", "info", "warning", "danger", "success"] = "primary",
                 button_size: Literal["mini", "small", "large"] = None,
                 plain: bool = False,
                 widget_id: str = None):
        self._widget_routes = {}

        self._text = text
        self._button_type = button_type
        self._plain = plain
        self._button_size = button_size

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            'text': self._text,
            'button_type': self._button_type,
            'plain': self._plain,
            'button_size': self._button_size,
        }

    def get_json_state(self):
        return None

