from typing import List
from supervisely.app.jinja2 import create_env
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget


class RadioTable(Widget):
    def __init__(
        self,
        columns: List[str],
        rows: List[List[str]],
        subtitles: dict = {},  # col_name -> subtitle
        column_formatters: dict = {},
        widget_id: str = None,
    ):
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

        super().__init__(widget_id=widget_id, file_path=__file__)

    def init_data(self):
        return {"header": self._header, "rows": self._frows}

    def init_state(self):
        return {"selectedRow": 0}

    def format_value(self, column_name: str, value):
        fn = self.column_formatters.get(column_name, self.default_formatter)
        return fn(value)

    def default_formatter(self, value):
        if value is None:
            return "-"
        return value
