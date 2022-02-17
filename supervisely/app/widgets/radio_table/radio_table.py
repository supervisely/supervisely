from typing import List, Dict
from pathlib import Path
import os
from black import format_cell
import markupsafe
from jinja2 import Environment
from supervisely.app.jinja2 import create_env

# from supervisely.api.api import Api
# from supervisely.app.content import DataJson, LastStateJson


class RadioTable:
    def __init__(
        self,
        widget_id: str,
        columns: List[str],
        rows: List[List[str]],
        subtitles: dict = {},  # col_name -> subtitle
        column_formatters: dict = {},
    ):
        self.widget_id = widget_id
        self.columns = columns
        self.rows = rows
        self.subtitles = subtitles
        self.column_formatters = column_formatters

        self._header = []
        for col in self.columns:
            self._header.append({"title": col, "subtitle": self.subtitles.get(col)})

        self._frows = []
        for idx, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"Row #{idx} length is {len(row)} != number of columns ({len(self.columns)})"
                )
            frow = []
            for col, val in zip(self.columns, row):
                frow.append(self.format_value(col, val))
            self._frows.append(frow)

    def format_value(self, column_name: str, value):
        fn = self.column_formatters.get(column_name, self.default_formatter)
        return fn(value)

    def default_formatter(self, value):
        if value is None:
            return "-"
        return value

    def init(self, data: dict, state: dict):
        data[self.widget_id] = {"header": self._header, "rows": self._frows}
        state[self.widget_id] = {"selectedRow": 0}

    def to_html(self):
        current_dir = Path(__file__).parent.absolute()
        jinja2_sly_env: Environment = create_env(current_dir)
        html = jinja2_sly_env.get_template("radio_table.html").render({"widget": self})
        return markupsafe.Markup(html)
