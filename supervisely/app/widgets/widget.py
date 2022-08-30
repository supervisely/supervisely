from pathlib import Path
from typing import Callable

from varname import varname
from jinja2 import Environment
import markupsafe
from supervisely.app.jinja2 import create_env
from supervisely.app.content import DataJson, StateJson

# from supervisely.app.fastapi import Jinja2Templates, Application
from supervisely.app.widgets_context import JinjaWidgets


class Widget:
    def __init__(self, widget_id: str = None, file_path: str = __file__):
        # self._sly_app = Application() - do not create app before it is created in main.py or in serving template
        self.widget_id = widget_id
        self._file_path = file_path
        if self.widget_id is None:
            try:
                self.widget_id = varname(frame=2)
            except Exception as e:
                self.widget_id = varname(frame=3)

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
        serialized_data = self.get_json_data()
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
        return markupsafe.Markup(html)

    def __html__(self):
        return self.to_html()


# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
# https://github.com/pwwang/python-varname
# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string/18425523#18425523
# https://ideone.com/ym3bkD
# https://github.com/pwwang/python-varname
# https://stackoverflow.com/questions/13034496/using-global-variables-between-files
# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
