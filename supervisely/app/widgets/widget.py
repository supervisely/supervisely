from pathlib import Path
from typing import Callable
import uuid

from varname import varname
from jinja2 import Environment
import markupsafe
from supervisely.app.jinja2 import create_env
from supervisely.app.content import DataJson, StateJson
from fastapi import FastAPI
from supervisely.app.fastapi import _MainServer
from supervisely.app.widgets_context import JinjaWidgets
from supervisely._utils import generate_free_name, rand_str


class Hidable:
    def __init__(self):
        self._hide = False

    def hide(self):
        self._hide = True
        DataJson()[self.widget_id]["hide"] = self._hide
        DataJson().send_changes()

    def show(self):
        self._hide = False
        DataJson()[self.widget_id]["hide"] = self._hide
        DataJson().send_changes()

    def get_json_data(self):
        return {"hide": self._hide}

    def get_json_state(self):
        raise {}


class Widget(Hidable):
    def __init__(self, widget_id: str = None, file_path: str = __file__):
        super().__init__()
        self._sly_app = _MainServer()
        self.widget_id = widget_id
        self._file_path = file_path
        if self.widget_id is None:
            try:
                self.widget_id = varname(frame=2)
            except Exception as e:
                try:
                    self.widget_id = varname(frame=3)
                except Exception as e:
                    self.widget_id = type(self).__name__ + rand_str(10)

        self._register()

    def _register(self):
        # get singletons
        data = DataJson()
        data.raise_for_key(self.widget_id)
        self.update_data()

        state = StateJson()
        state.raise_for_key(self.widget_id)
        self.update_state(state=state)

        JinjaWidgets().context[self.widget_id] = self
        # templates = Jinja2Templates()
        # templates.context_widgets[self.widget_id] = self

    def get_json_data(self):
        raise NotImplementedError()

    def get_json_state(self):
        raise NotImplementedError()

    def update_state(self, state=None):
        serialized_state = self.get_json_state()
        if serialized_state is not None:
            if state is None:
                state = StateJson()
            state.setdefault(self.widget_id, {}).update(serialized_state)

    def update_data(self):
        data = DataJson()

        widget_data = self.get_json_data()
        if widget_data is None:
            widget_data = {}
        hidable_data = super().get_json_data()

        serialized_data = {**widget_data, **hidable_data}
        if serialized_data is not None:
            data.setdefault(self.widget_id, {}).update(serialized_data)

    def get_route_path(self, route: str) -> str:
        return f"/{self.widget_id}/{route}"

    def add_route(self, app, route):
        def decorator(f):
            existing_cb = DataJson()[self.widget_id].get("widget_routes", {}).get(route)
            if existing_cb is not None:
                raise Exception(
                    f"Route [{route}] already attached to function with name: {existing_cb}"
                )

            app.add_api_route(f"/{self.widget_id}/{route}", f, methods=["POST"])
            DataJson()[self.widget_id].setdefault("widget_routes", {})[
                route
            ] = f.__name__

            self.update_data()

        return decorator

    def to_html(self):
        current_dir = Path(self._file_path).parent.absolute()
        jinja2_sly_env: Environment = create_env(current_dir)
        html = jinja2_sly_env.get_template("template.html").render({"widget": self})

        # hidable v-if
        # @TODO: reimplement with jinja2 templating
        res = f'<div v-if="!data.{self.widget_id}.hide">{html}</div>'
        return markupsafe.Markup(res)

    def __html__(self):
        res = self.to_html()
        return res


# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
# https://github.com/pwwang/python-varname
# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string/18425523#18425523
# https://ideone.com/ym3bkD
# https://github.com/pwwang/python-varname
# https://stackoverflow.com/questions/13034496/using-global-variables-between-files
# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
