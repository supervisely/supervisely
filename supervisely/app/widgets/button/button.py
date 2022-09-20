from functools import wraps

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class Button(Widget):
    class Routes:
        CLICK = "button_clicked_cb"

    def __init__(
        self,
        text: str = "Button",
        button_type: Literal["primary", "info", "warning", "danger", "success", "text"] = "primary",
        button_size: Literal["mini", "small", "large"] = None,
        plain: bool = False,
        show_loading: bool = True,
        icon: str = None,  # for example "zmdi zmdi-play" from http://zavoloklom.github.io/material-design-iconic-font/icons.html
        icon_gap: int = 5,
        widget_id=None,
    ):
        self._widget_routes = {}

        self._text = text
        self._button_type = button_type
        self._button_size = button_size
        self._plain = plain
        self._icon_gap = icon_gap
        if icon is None:
            self._icon = ""
        else:
            self._icon = f'<i class="{icon}" style="margin-right: {icon_gap}px"></i>'

        self._loading = False
        self._disabled = False
        self._show_loading = show_loading

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "text": self._text,
            "button_type": self._button_type,
            "plain": self._plain,
            "button_size": self._button_size,
            "loading": self._loading,
            "disabled": self._disabled,
            "icon": self._icon,
        }

    def get_json_state(self):
        return None

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value
        DataJson()[self.widget_id]["text"] = self._text
        DataJson().send_changes()

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    @property
    def show_loading(self):
        return self._show_loading

    @property
    def disabled(self):
        return self._disabled

    @disabled.setter
    def disabled(self, value):
        self._disabled = value
        DataJson()[self.widget_id]["disabled"] = self._disabled

    def click(self, func):
        # from fastapi import Request

        route_path = self.get_route_path(Button.Routes.CLICK)
        server = self._sly_app.get_server()

        @server.post(route_path)
        def _click():
            # maybe work with headers and store some values there r: Request
            if self.show_loading:
                self.loading = True
            try:
                func()
            except Exception as e:
                if self.show_loading and self.loading:
                    self.loading = False
                raise e
            if self.show_loading:
                self.loading = False

        return _click
