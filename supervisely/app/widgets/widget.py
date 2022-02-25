import os
from pathlib import Path
from varname import varname
from jinja2 import Environment
import markupsafe
from supervisely.app.jinja2 import create_env
from supervisely.app.content import DataJson, StateJson
from supervisely.app.fastapi import Jinja2Templates


class Widget:
    def __init__(self, widget_id: str = None, file_path: str = __file__):
        self.widget_id = widget_id
        self._file_path = file_path
        if self.widget_id is None:
            self.widget_id = varname(frame=2)
        self._register()

    def _register(self):
        # get singletons
        data = DataJson()
        state = StateJson()
        data.raise_for_key(self.widget_id)
        data[self.widget_id] = self.init_data()
        state.raise_for_key(self.widget_id)
        state[self.widget_id] = self.init_state()
        templates = Jinja2Templates()
        templates.context_widgets[self.widget_id] = self

    def init_data(self):
        raise NotImplementedError()

    def init_state(self):
        raise NotImplementedError()

    def to_html(self):
        current_dir = Path(self._file_path).parent.absolute()
        jinja2_sly_env: Environment = create_env(current_dir)
        html = jinja2_sly_env.get_template("template.html").render({"widget": self})
        return markupsafe.Markup(html)


# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
# https://github.com/pwwang/python-varname
# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string/18425523#18425523
# https://ideone.com/ym3bkD
# https://github.com/pwwang/python-varname
# https://stackoverflow.com/questions/13034496/using-global-variables-between-files
# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
