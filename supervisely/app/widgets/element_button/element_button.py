import copy
from typing import Literal

import fastapi
from varname import varname

from supervisely.app.widgets import Widget


class ElementButton(Widget):
    class Routes:
        def __init__(self,
                     app: fastapi.FastAPI,
                     button_clicked: object = None):
            self.app = app
            self.routes = {'button_clicked_cb': button_clicked}

    def __init__(self,
                 widget_routes: Routes,
                 text: str = 'Element Button',
                 button_type: Literal["primary", "info", "warning", "danger", "success"] = "primary",
                 button_size: Literal["mini", "small", "large"] = None,
                 plain: bool = False,
                 widget_id: str = None):
        self.widget_id = varname(frame=1) if widget_id is None else widget_id

        self._widget_routes = widget_routes

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

    def add_widget_routes(self, routes: Routes):
        if routes is not None:
            for route_name, route_cb in routes.routes.items():
                if callable(route_cb):
                    routes.app.add_api_route(f'/{self.widget_id}/{route_name}', route_cb, methods=["POST"])
