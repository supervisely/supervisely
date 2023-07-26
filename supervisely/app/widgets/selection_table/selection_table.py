from typing import List, Dict, Union
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget


class SelectionTable(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        columns: List[str],
        rows: List[Dict[str, Union[str, int]]] = [],
        identity_field: str = None,
        columns_widths: Dict[str, int] = None,
        enable_selection: bool = True,
        widget_id: str = None,
    ):
        self._changes_handled = False

        if enable_selection is True:
            if identity_field is None:
                raise ValueError(f"Identity field should be specified if enable_selection is True")
            if identity_field not in columns:
                raise ValueError(
                    f"Identity field '{identity_field}' should be one of the columns (unique identifier for each row)"
                )

        if columns_widths is not None:
            for col in columns:
                if col not in columns_widths:
                    columns_widths[col] = None
        else:
            columns_widths = {col: None for col in columns}

        self._validate_data(rows, columns)

        # table data
        self._columns = columns
        self._rows = rows
        self._selected_rows = None
        self._identities = None

        # table options
        self._columns_widths = columns_widths
        self._enable_selection = enable_selection
        self._identity_field = identity_field

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "rows": self._rows,
            "columns": self._columns,
            "options": {
                "columnsWidths": self._columns_widths,
                "identityField": self._identity_field,
                "enableSelection": self._enable_selection,
            },
        }

    def get_json_state(self):
        return {"selectedRows": self._selected_rows}

    def _update_identities(self):
        self._rows = DataJson()[self.widget_id]["rows"]
        self._identities = [row[self._identity_field] for row in self._rows]

    def _validate_data(self, rows: List[Dict[str, Union[str, int]]], columns: List[str]):
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Row should be a dict, got: {type(row)}")
            for col in columns:
                if col not in row:
                    raise ValueError(f"Column '{col}' is not found in row: {row}")

    def _validate_new_rows(self, rows: List[Dict[str, Union[str, int]]], columns: List[str]):
        self._validate_data(rows, columns)
        self._update_identities()
        
        duplicate_identities = set([row[self._identity_field] for row in rows]) & set(self._identities)
        if len(duplicate_identities) > 0:
            raise ValueError(f"Rows with identities {duplicate_identities} already exist")

    @property
    def columns(self) -> List[str]:
        self._columns = DataJson()[self.widget_id]["columns"]
        return self._columns

    def add_row(self, row: Dict[str, Union[str, int]]) -> None:
        self._validate_new_rows([row], self.columns)
        self._rows.append(row)
        DataJson()[self.widget_id]["rows"] = self._rows
        DataJson().send_changes()

    def add_rows(self, rows: List[Dict[str, Union[str, int]]]) -> None:
        self._validate_new_rows(rows, self.columns)
        self._rows.extend(rows)
        DataJson()[self.widget_id]["rows"] = self._rows
        DataJson().send_changes()

    def remove_row(self, identity: Union[str, int]) -> None:
        self._update_identities()
        if identity not in self._identities:
            return
        self._rows = [row for row in self._rows if row[self._identity_field] != identity]
        DataJson()[self.widget_id]["rows"] = self._rows
        DataJson().send_changes()
        self._selected_rows = StateJson()[self.widget_id]["selectedRows"]
        self._selected_rows = [row for row in self._selected_rows if row[self._identity_field] != identity]
        StateJson()[self.widget_id]["selectedRows"] = self._selected_rows
        StateJson().send_changes()


    @property
    def rows(self):
        self._rows = DataJson()[self.widget_id]["rows"]
        return self._rows

    def get_selected_rows(self) -> List[dict]:
        self._selected_rows = StateJson()[self.widget_id]["selectedRows"]
        return self._selected_rows

    def get_selected_identities(self) -> List[Union[str, int]]:
        self._selected_rows = StateJson()[self.widget_id]["selectedRows"]
        indentity_fields = [row[self._identity_field] for row in self._selected_rows]
        return indentity_fields

    def value_changed(self, func):
        route_path = self.get_route_path(SelectionTable.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            res = self.get_selected_rows()
            func(res)

        return _value_changed
