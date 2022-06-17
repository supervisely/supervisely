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
        self._rows = rows
        self.subtitles = subtitles
        self.column_formatters = column_formatters

        self._header = []
        for col in self.columns:
            self._header.append({"title": col, "subtitle": self.subtitles.get(col)})

        self._frows = []

        super().__init__(widget_id=widget_id, file_path=__file__)

        self.rows = rows

    def get_json_data(self):
        return {
            "header": self._header,
            "frows": self._frows,
            "raw_rows_data": self.rows,
        }

    def get_json_state(self):
        return {"selectedRow": 0}

    def format_value(self, column_name: str, value):
        fn = self.column_formatters.get(column_name, self.default_formatter)
        return fn(f"data.{self.widget_id}.raw_rows_data[params.ridx][params.vidx]")

    def default_formatter(self, value):
        if value is None:
            return "-"
        return "<div> {{{{ data.{}.raw_rows_data[params.ridx][params.vidx] }}}} </div>".format(
            self.widget_id
        )

    def _update_frows(self):
        self._frows = []
        for idx, row in enumerate(self._rows):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"Row #{idx} length is {len(row)} != number of columns ({len(self.columns)})"
                )
            frow = []
            for col, val in zip(self.columns, row):
                frow.append(self.format_value(col, val))
            self._frows.append(frow)

    def get_selected_row(self, state):
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        if widget_actual_state is not None and widget_actual_data is not None:
            selected_row_index = widget_actual_state["selectedRow"]
            return self.rows[selected_row_index]

    @property
    def rows(self):
        return self._rows

    @rows.setter
    def rows(self, value):
        self._rows = value
        self._update_frows()
        DataJson()[self.widget_id]["frows"] = self._frows
        DataJson()[self.widget_id]["raw_rows_data"] = self._rows
