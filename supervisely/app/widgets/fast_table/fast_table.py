import copy
import traceback
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from supervisely.app import DataJson
from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger


class EventLinstenerError(Exception):
    def __init__(self, message="An exception occurred due to conflicting event listeners."):
        self.message = message
        super().__init__(self.message)


class FastTable(Widget):
    """FastTable widget in Supervisely allows for displaying and manipulating data of various
    dataset statistics and processing it on the server side.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/tables/fasttable>`_
        (including screenshots and examples).

    :param data: dataset or project data in different formats:
    :type data: Union[pd.DataFrame, List]
    :param columns: List of column names
    :type columns: List, optional
    :param columns_options: List of dicts with options for each column
    :type columns_options: List[dict], optional
    :param project_meta: Project meta information
    :type project_meta: Union[ProjectMeta, dict], optional
    :param fixed_columns: Number of fixed columns
    :type fixed_columns: Literal[1], optional
    :param page_size: Number of rows per page
    :type page_size: int, optional
    :param sort_column_idx: Index of the column to sort by
    :type sort_column_idx: int, optional
    :param sort_order: Sorting order
    :type sort_order: Literal["asc", "desc"], optional
    :param width: Width of the widget, e.g. "200px", or "100%"
    :type width: str, optional
    :param widget_id: Unique widget identifier.
    :type widget_id: str, optional
    :param show_header: Whether to show table header
    :type show_header: bool, optional
    :param is_radio: Enable radio button selection mode (single row selection)
    :type is_radio: bool, optional
    :param is_selectable: Enable multiple row selection
    :type is_selectable: bool, optional
    :param header_left_content: Widget to display in the left side of the header
    :type header_left_content: Widget, optional
    :param header_right_content: Widget to display in the right side of the header
    :type header_right_content: Widget, optional
    :param max_selected_rows: Maximum number of rows that can be selected
    :type max_selected_rows: int, optional
    :param search_position: Position of the search input ("left" or "right")
    :type search_position: Literal["left", "right"], optional


    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import FastTable

        data = [["apple", "21"], ["banana", "15"]]
        columns = ["Class", "Items"]
        dataframe = pd.DataFrame(data=data, columns=columns)
        columns_options = [
            { "type": "class"},
            { "maxValue": 21, "postfix": "pcs", "tooltip": "description text", "subtitle": "boxes" }
        ]

        meta_path = "meta.json"  # define file path
        with open(meta_path, "r") as json_file:
            meta = json.load(json_file)

        fast_table = FastTable(
            data=dataframe,
            project_meta=meta,
            columns_options=columns_options,
        )
    """

    class Routes:
        SELECTION_CHANGED = "selection_changed_cb"
        ROW_CLICKED = "row_clicked_cb"
        CELL_CLICKED = "cell_clicked_cb"
        UPDATE_DATA = "update_data_cb"

    class ClickedRow:
        def __init__(
            self,
            row: List,
            row_index: int = None,
        ):
            self.row = row
            self.row_index = row_index

    class ClickedCell:
        def __init__(
            self,
            row: List,
            column_index: int,
            row_index: int = None,
            column_name: int = None,
            column_value: int = None,
        ):
            self.row = row
            self.column_index = column_index
            self.row_index = row_index
            self.column_name = column_name
            self.column_value = column_value

    class ColumnData:
        def __init__(self, name, is_widget=False, widget: Widget = None):
            self.name = name
            self.is_widget = is_widget
            self.widget = widget

        @property
        def widget_html(self):
            html = self.widget.to_html()
            html = html.replace(f".{self.widget.widget_id}", "[JSON.parse(cellValue).widget_id]")
            html = html.replace(
                f"/{self.widget.widget_id}", "/' + JSON.parse(cellValue).widget_id + '"
            )
            if hasattr(self.widget, "_widgets"):
                for i, widget in enumerate(self.widget._widgets):
                    html = html.replace(
                        f".{widget.widget_id}", f"[JSON.parse(cellValue).widgets[{i}]]"
                    )
                    html = html.replace(
                        f"/{widget.widget_id}", f"/' + JSON.parse(cellValue).widgets[{i}] + '"
                    )
            return html

    def __init__(
        self,
        data: Optional[Union[pd.DataFrame, List]] = None,
        columns: Optional[List] = None,
        columns_options: Optional[List[dict]] = None,
        project_meta: Optional[Union[ProjectMeta, dict]] = None,
        fixed_columns: Optional[Literal[1]] = None,
        page_size: Optional[int] = 10,
        sort_column_idx: int = None,
        sort_order: Optional[Literal["asc", "desc"]] = None,
        width: Optional[str] = "auto",
        widget_id: Optional[str] = None,
        show_header: bool = True,
        is_radio: bool = False,
        is_selectable: bool = False,
        header_left_content: Optional[Widget] = None,
        header_right_content: Optional[Widget] = None,
        max_selected_rows: Optional[int] = None,
        search_position: Optional[Literal["left", "right"]] = None,
    ):
        self._supported_types = tuple([pd.DataFrame, list, type(None)])
        self._row_click_handled = False
        self._cell_click_handled = False
        self._selection_changed_handled = False
        self._columns = columns
        self._columns_data = []
        if columns is None:
            self._columns_first_idx = None
        else:
            self._columns_first_idx = []
            for col in columns:
                if isinstance(col, str):
                    self._columns_first_idx.append(col)
                    self._columns_data.append(self.ColumnData(name=col))
                elif isinstance(col, tuple):
                    self._columns_first_idx.append(col[0])
                    self._columns_data.append(
                        self.ColumnData(name=col[0], is_widget=True, widget=col[1])
                    )
                else:
                    raise TypeError(f"Column name must be a string or a tuple, got {type(col)}")

        self._columns_options = columns_options
        self._sorted_data = None
        self._filtered_data = None
        self._searched_data = None
        self._active_page = 1
        self._width = width
        self._selected_rows = []
        self._selected_cell = None
        self._clicked_row = None
        self._is_row_clickable = False
        self._is_cell_clickable = False
        self._search_str = ""
        self._show_header = show_header
        self._project_meta = self._unpack_project_meta(project_meta)
        self._header_left_content = header_left_content
        self._header_right_content = header_right_content
        self._max_selected_rows = max_selected_rows
        acceptable_search_positions = ["left", "right"]
        self._search_position = search_position if search_position in acceptable_search_positions else "left"

        # table_options
        self._page_size = page_size
        self._fix_columns = self._validate_fix_columns_value(fixed_columns)
        self._sort_column_idx = sort_column_idx
        self._sort_order = sort_order
        self._validate_sort_attrs()
        self._is_radio = is_radio
        self._is_selectable = is_selectable
        self._search_function = self._default_search_function
        self._sort_function = self._default_sort_function
        self._filter_function = self._default_filter_function
        self._filter_value = None

        # to avoid errors with the duplicated names in columns
        self._multi_idx_columns = None

        # prepare source_data
        self._validate_input_data(data)
        self._source_data = self._prepare_input_data(data)

        # Initialize filtered and searched data for proper initialization
        self._filtered_data = self._filter(self._filter_value)
        self._searched_data = self._search(self._search_str)
        self._sorted_data = self._sort_table_data(self._searched_data)

        # prepare parsed_source_data, sliced_data, parsed_active_data
        (
            self._parsed_source_data,
            self._sliced_data,
            self._parsed_active_data,
        ) = self._prepare_working_data()

        self._rows_total = len(self._parsed_source_data["data"])

        if self._is_radio and self._rows_total > 0:
            self._selected_rows = [self._parsed_source_data["data"][0]]

        super().__init__(widget_id=widget_id, file_path=__file__)

        script_path = "./sly/css/app/widgets/fast_table/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

        filter_changed_route_path = self.get_route_path(FastTable.Routes.UPDATE_DATA)
        server = self._sly_app.get_server()

        @server.post(filter_changed_route_path)
        def _filter_changed_handler():
            self._refresh()

    def _refresh(self):
        # TODO sort widgets
        self._active_page = StateJson()[self.widget_id]["page"]
        self._sort_order = StateJson()[self.widget_id]["sort"]["order"]
        self._sort_column_idx = StateJson()[self.widget_id]["sort"]["column"]
        search_value = StateJson()[self.widget_id]["search"]
        self._filtered_data = self._filter(self._filter_value)
        self._searched_data = self._search(search_value)
        self._rows_total = len(self._searched_data)

        # if active page is greater than the number of pages (e.g. after filtering)
        max_page = (self._rows_total - 1) // self._page_size + 1
        if (self._rows_total > 0 and self._active_page == 0) or self._active_page > max_page:
            self._active_page = 1
            StateJson()[self.widget_id]["page"] = self._active_page

        self._sorted_data = self._sort_table_data(self._searched_data)
        self._sliced_data = self._slice_table_data(self._sorted_data, actual_page=self._active_page)
        self._parsed_active_data = self._unpack_pandas_table_data(self._sliced_data)
        StateJson().send_changes()
        DataJson()[self.widget_id]["data"] = list(self._parsed_active_data["data"])
        DataJson()[self.widget_id]["total"] = self._rows_total
        DataJson().send_changes()
        StateJson()["reactToChanges"] = True
        StateJson().send_changes()

    def get_json_data(self) -> Dict[str, Any]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.
        Dictionary contains the following fields:
            - data: table data
            - columns: list of column names
            - projectMeta: project meta information
            - columnsOptions: list of dicts with options for each column
            - total: total number of rows
            - options: table options with the following fields:
                - isRowClickable: whether rows are clickable
                - isCellClickable: whether cells are clickable
                - fixColumns: number of fixed columns
                - isRadio: whether radio button selection mode is enabled
                - isRowSelectable: whether multiple row selection is enabled
                - maxSelectedRows: maximum number of rows that can be selected
                - searchPosition: position of the search input ("left" or "right")
            - pageSize: number of rows per page
            - showHeader: whether to show table header
            - selectionChangedHandled: whether selection changed event listener is set

        :return: Dictionary with widget data
        :rtype: Dict[str, Any]
        """
        return {
            "data": list(self._parsed_active_data["data"]),
            "columns": self._parsed_source_data["columns"],
            "projectMeta": self._project_meta,
            "columnsOptions": self._columns_options,
            "total": self._rows_total,
            "options": {
                "isRowClickable": self._is_row_clickable,
                "isCellClickable": self._is_cell_clickable,
                "fixColumns": self._fix_columns,
                "isRadio": self._is_radio,
                "isRowSelectable": self._is_selectable,
                "maxSelectedRows": self._max_selected_rows,
                "searchPosition": self._search_position,
            },
            "pageSize": self._page_size,
            "showHeader": self._show_header,
            "selectionChangedHandled": self._selection_changed_handled,
        }

    def get_json_state(self) -> Dict[str, Any]:
        """Returns dictionary with widget state.
        Dictionary contains the following fields:
            - search: search string
            - selectedRows: selected rows
            - selectedCell: selected cell
            - clickedRow: clicked row
            - page: active page
            - sort: sorting options with the following fields:
                - column: index of the column to sort by
                - order: sorting order

        :return: Dictionary with widget state
        :rtype: Dict[str, Any]
        """
        return {
            "search": self._search_str,
            "selectedRows": self._selected_rows,
            "selectedCell": self._selected_cell,
            "clickedRow": self._clicked_row,
            "page": self._active_page,
            "sort": {
                "column": self._sort_column_idx,
                "order": self._sort_order,
            },
        }

    @property
    def fixed_columns_num(self) -> int:
        """Returns number of fixed columns.

        :return: Number of fixed columns
        :rtype: int
        """
        return self._fix_columns

    @fixed_columns_num.setter
    def fixed_columns_num(self, value: int) -> None:
        """Sets number of fixed columns.

        :param value: Number of fixed columns
        :type value: int
        """
        self._fix_columns = self._validate_fix_columns_value(value)
        DataJson()[self.widget_id]["options"]["fixColumns"] = self._fix_columns

    @property
    def project_meta(self) -> Dict[str, Any]:
        """Returns project meta information.

        :return: Project meta information
        :rtype: Dict[str, Any]
        """
        return self._project_meta

    @project_meta.setter
    def project_meta(self, meta: Union[ProjectMeta, Dict]) -> None:
        """Sets project meta information.

        :param meta: Project meta information
        :type meta: Union[ProjectMeta, Dict]
        """
        self._project_meta = self._unpack_project_meta(meta)
        DataJson()[self.widget_id]["projectMeta"] = self._project_meta

    @property
    def page_size(self) -> int:
        return self._page_size

    @page_size.setter
    def page_size(self, size: int):
        self._page_size = size
        DataJson()[self.widget_id]["pageSize"] = self._page_size

    def set_sort(
        self, func: Callable[[pd.DataFrame, int, Optional[Literal["asc", "desc"]]], pd.DataFrame]
    ) -> None:
        """Sets custom sort function for the table.

        :param func: Custom sort function
        :type func: Callable[[pd.DataFrame, int, Optional[Literal["asc", "desc"]]], pd.DataFrame]
        """
        self._sort_function = func

    def set_search(self, func: Callable[[pd.DataFrame, str], pd.DataFrame]) -> None:
        """Sets custom search function for the table.

        :param func: Custom search function
        :type func: Callable[[pd.DataFrame, str], pd.DataFrame]
        """
        self._search_function = func

    def set_filter(self, filter_function: Callable[[pd.DataFrame, Any], pd.DataFrame]) -> None:
        """Sets a custom filter function for the table.
        first argument is a DataFrame, second argument is a filter value.

        :param filter_function: Custom filter function
        :type filter_function: Callable[[pd.DataFrame, Any], pd.DataFrame]
        """
        if filter_function is None:
            filter_function = self._default_filter_function
        self._filter_function = filter_function

    def read_json(self, data: Dict, meta: Dict = None, custom_columns: Optional[List[Union[str, tuple]]] = None) -> None:
        """Replace table data with options and project meta in the widget

        More about options in `Developer Portal <https://developer.supervisely.com/app-development/widgets/tables/fasttable#read_json>`_

        :param data: Table data with options:
            - data: table data
            - columns: list of column names
            - projectMeta: project meta information - if provided
            - columnsOptions: list of dicts with options for each column
            - total: total number of rows
            - options: table options with the following fields:
                - isRowClickable: whether rows are clickable
                - isCellClickable: whether cells are clickable
                - fixColumns: number of fixed columns
                - isRadio: whether radio button selection mode is enabled
                - isRowSelectable: whether multiple row selection is enabled
                - maxSelectedRows: maximum number of rows that can be selected
                - searchPosition: position of the search input ("left" or "right")
            - pageSize: number of rows per page
            - showHeader: whether to show table header
            - selectionChangedHandled: whether selection changed event listener is set

        :type data: dict
        :param meta: Project meta information
        :type meta: dict
        :param custom_columns: List of column names. Can include widgets as tuples (column_name, widget)
        :type custom_columns: List[Union[str, tuple]], optional

        Example of data dict:
        .. code-block:: python

            data = {
                "data": [["apple", "21"], ["banana", "15"]],
                "columns": ["Class", "Items"],
                "columnsOptions": [
                    { "type": "class"},
                    { "maxValue": 21, "postfix": "pcs", "tooltip": "description text", "subtitle": "boxes" }
                ],
                "options": {
                    "isRowClickable": True,
                    "isCellClickable": True,
                    "fixColumns": 1,
                    "isRadio": False,
                    "isRowSelectable": True,
                    "maxSelectedRows": 5,
                    "searchPosition": "right",
                    "sort": {"column": 0, "order": "asc"},
                },
            }
        """
        self._columns_options = self._prepare_json_data(data, "columnsOptions")
        self._read_custom_columns(custom_columns)
        if not self._columns_first_idx:
            self._columns_first_idx = self._prepare_json_data(data, "columns")
        self._table_options = self._prepare_json_data(data, "options")
        self._project_meta = self._unpack_project_meta(meta)
        table_data = data.get("data", None)
        self._validate_input_data(table_data)
        self._source_data = self._prepare_input_data(table_data)

        init_options = DataJson()[self.widget_id]["options"]
        init_options.update(self._table_options)
        sort = init_options.pop("sort", {"column": None, "order": None})
        self._active_page = 1
        self._sort_column_idx = sort.get("column", None)
        if self._sort_column_idx is not None and self._sort_column_idx > len(self._columns_first_idx) - 1:
            self._sort_column_idx = None
        self._sort_order = sort.get("order", None)
        self._page_size = init_options.pop("pageSize", 10)

        # Apply sorting before preparing working data
        self._sorted_data = self._sort_table_data(self._source_data)
        self._sliced_data = self._slice_table_data(self._sorted_data, actual_page=self._active_page)
        self._parsed_active_data = self._unpack_pandas_table_data(self._sliced_data)
        self._parsed_source_data = self._unpack_pandas_table_data(self._source_data)
        self._rows_total = len(self._parsed_source_data["data"]) 
        DataJson()[self.widget_id]["data"] = list(self._parsed_active_data["data"])
        DataJson()[self.widget_id]["columns"] = self._parsed_active_data["columns"]
        DataJson()[self.widget_id]["columnsOptions"] = self._columns_options
        DataJson()[self.widget_id]["options"] = init_options
        DataJson()[self.widget_id]["total"] = len(self._source_data)
        DataJson()[self.widget_id]["pageSize"] = self._page_size
        DataJson()[self.widget_id]["projectMeta"] = self._project_meta
        StateJson()[self.widget_id]["sort"]["column"] = self._sort_column_idx
        StateJson()[self.widget_id]["sort"]["order"] = self._sort_order
        StateJson()[self.widget_id]["page"] = self._active_page
        StateJson()[self.widget_id]["selectedRows"] = []
        StateJson()[self.widget_id]["selectedCell"] = None
        self._maybe_update_selected_row()
        self._validate_sort_attrs()
        DataJson().send_changes()
        StateJson().send_changes()

    def read_pandas(self, data: pd.DataFrame) -> None:
        """Replace table data (rows and columns) in the widget.

        :param data: Table data
        :type data: pd.DataFrame
        """
        self._source_data = self._prepare_input_data(data)
        self._sorted_data = self._sort_table_data(self._source_data)
        self._sliced_data = self._slice_table_data(self._sorted_data)
        self._parsed_active_data = self._unpack_pandas_table_data(self._sliced_data)
        self._parsed_source_data = self._unpack_pandas_table_data(self._source_data)
        self._rows_total = len(self._parsed_source_data["data"])
        DataJson()[self.widget_id]["data"] = list(self._parsed_active_data["data"])
        DataJson()[self.widget_id]["columns"] = self._parsed_active_data["columns"]
        DataJson()[self.widget_id]["total"] = len(self._source_data)
        DataJson().send_changes()
        self._active_page = 1
        StateJson()[self.widget_id]["page"] = self._active_page
        StateJson().send_changes()
        self.clear_selection()

    def to_json(self, active_page: Optional[bool] = False) -> Dict[str, Any]:
        """Export table data with current options as dict.

        Dictionary contains the following fields:
            - data: table data
            - columns: list of column names
            - projectMeta: project meta information - if provided
            - columnsOptions: list of dicts with options for each column
            - total: total number of rows
            - options: table options with the following fields:
                - isRowClickable: whether rows are clickable
                - isCellClickable: whether cells are clickable
                - fixColumns: number of fixed columns
                - isRadio: whether radio button selection mode is enabled
                - isRowSelectable: whether multiple row selection is enabled
                - maxSelectedRows: maximum number of rows that can be selected
                - searchPosition: position of the search input ("left" or "right")
            - pageSize: number of rows per page
            - showHeader: whether to show table header
            - selectionChangedHandled: whether selection changed event listener is set

        :param active_page: Specifies the size of the data to be exported. If True - returns only the active page of the table
        :type active_page: Optional[bool]
        :return: Table data with current options
        :rtype: dict
        """
        if active_page is True:
            temp_parsed_data = [d["items"] for d in self._parsed_active_data["data"]]
        else:
            temp_parsed_data = [d["items"] for d in self._parsed_source_data["data"]]
        widget_data = {}
        widget_data["data"] = temp_parsed_data
        widget_data["columns"] = DataJson()[self.widget_id]["columns"]
        widget_data["options"] = copy.deepcopy(DataJson()[self.widget_id]["options"])
        widget_data["columnsOptions"] = DataJson()[self.widget_id]["columnsOptions"]
        sort = copy.deepcopy(StateJson()[self.widget_id]["sort"])
        sort["columnIndex"] = sort.get("column", None)
        sort.pop("column", None)
        widget_data["options"]["sort"] = sort
        return widget_data

    def to_pandas(self, active_page=False) -> pd.DataFrame:
        """Export only table data (rows and columns) as Pandas Dataframe.

        :param active_page: Specifies the size of the data to be exported. If True - returns only the active page of the table
        :type active_page: bool
        :return: Table data
        :rtype: pd.DataFrame
        """
        if active_page is True:
            # Return sliced data directly from source to preserve None/NaN values
            packed_data = self._sliced_data.copy()
            # Reset column names to first level only
            if isinstance(packed_data.columns, pd.MultiIndex):
                packed_data.columns = packed_data.columns.get_level_values("first")
        else:
            # Return source data directly to preserve None/NaN values
            packed_data = self._source_data.copy()
            # Reset column names to first level only
            if isinstance(packed_data.columns, pd.MultiIndex):
                packed_data.columns = packed_data.columns.get_level_values("first")
        return packed_data

    def clear_selection(self) -> None:
        """Clears the selection of the table."""
        StateJson()[self.widget_id]["selectedRows"] = []
        StateJson()[self.widget_id]["selectedCell"] = None
        StateJson().send_changes()
        self._maybe_update_selected_row()

    def get_selected_row(self) -> ClickedRow:
        """Returns the selected row.

        :return: Selected row
        :rtype: ClickedRow
        """
        if self._is_radio or self._is_selectable:
            selected_rows = StateJson()[self.widget_id]["selectedRows"]
            if selected_rows is None:
                return None
            if len(selected_rows) == 0:
                return None
            if len(selected_rows) > 1:
                raise ValueError(
                    "Multiple rows selected. Use get_selected_rows() method to get all selected rows."
                )
            row = selected_rows[0]
            row_index = row["idx"]
            row = row.get("row", row.get("items", None))
            if row_index is None or row is None:
                return None
            return self.ClickedRow(row, row_index)
        return self.get_clicked_row()

    def get_selected_rows(self) -> List[ClickedRow]:
        if self._is_radio or self._is_selectable:
            selected_rows = StateJson()[self.widget_id]["selectedRows"]
            rows = []
            for row in selected_rows:
                row_index = row["idx"]
                if row_index is None:
                    continue
                # Get original data from source_data to preserve None/NaN values
                try:
                    row_data = self._source_data.loc[row_index].values.tolist()
                except (KeyError, IndexError):
                    continue
                rows.append(self.ClickedRow(row_data, row_index))
            return rows
        return [self.get_clicked_row()]

    def get_clicked_row(self) -> ClickedRow:
        clicked_row = StateJson()[self.widget_id]["clickedRow"]
        if clicked_row is None:
            return None
        row_index = clicked_row["idx"]
        if row_index is None:
            return None
        # Get original data from source_data to preserve None/NaN values
        try:
            row = self._source_data.loc[row_index].values.tolist()
        except (KeyError, IndexError):
            return None
        return self.ClickedRow(row, row_index)

    def get_clicked_cell(self) -> ClickedCell:
        """Returns the selected cell.

        :return: Selected cell
        :rtype: ClickedCell
        """
        cell_data = StateJson()[self.widget_id]["selectedCell"]
        if cell_data is None:
            return None
        row_index = cell_data["idx"]
        column_index = cell_data["column"]
        if column_index is None or row_index is None:
            return None
        column_name = self._columns_first_idx[column_index]
        # Get original data from source_data to preserve None/NaN values
        try:
            row = self._source_data.loc[row_index].values.tolist()
            column_value = row[column_index]
        except (KeyError, IndexError):
            return None
        return self.ClickedCell(row, column_index, row_index, column_name, column_value)

    def get_selected_cell(self) -> ClickedCell:
        """Alias for get_clicked_cell method.
        Will be removed in future versions.
        """
        return self.get_clicked_cell()

    def _maybe_update_selected_row(self) -> None:
        if self._is_radio:
            if self._rows_total != 0:
                self.select_row(0)
            else:
                self._selected_rows = []
                StateJson()[self.widget_id]["selectedRows"] = self._selected_rows
                StateJson().send_changes()
            return
        if not self._selected_rows:
            return
        if self._rows_total == 0:
            self._selected_rows = []
            StateJson()[self.widget_id]["selectedRows"] = self._selected_rows
            StateJson().send_changes()
            return
        if self._is_selectable:
            updated_selected_rows = []
            for row in self._parsed_source_data["data"]:
                items = row.get("items", row.get("row", None))
                if items is not None:
                    for selected_row in self._selected_rows:
                        if selected_row.get("row", selected_row.get("items", None)) == items:
                            updated_selected_rows.append(row)
            self._selected_rows = updated_selected_rows
            StateJson()[self.widget_id]["selectedRows"] = self._selected_rows
            StateJson().send_changes()

    def insert_row(self, row: List, index: Optional[int] = -1) -> None:
        """Inserts a row into the table to the specified position.

        :param row: Row to insert
        :type row: List
        :param index: Index of the row to insert
        :type index: Optional[int]
        """
        self._validate_table_sizes(row)
        self._validate_row_values_types(row)
        index = len(self._source_data) if index > len(self._source_data) or index < 0 else index

        self._source_data = pd.concat(
            [
                self._source_data.iloc[:index],
                pd.DataFrame([row], columns=self._source_data.columns),
                self._source_data.iloc[index:],
            ]
        ).reset_index(drop=True)
        (
            self._parsed_source_data,
            self._sliced_data,
            self._parsed_active_data,
        ) = self._prepare_working_data()
        self._rows_total = len(self._parsed_source_data["data"])
        DataJson()[self.widget_id]["data"] = list(self._parsed_active_data["data"])
        DataJson()[self.widget_id]["total"] = self._rows_total
        DataJson().send_changes()
        self._maybe_update_selected_row()

    def add_rows(self, rows: List):
        for row in rows:
            self._validate_table_sizes(row)
            self._validate_row_values_types(row)
        self._source_data = pd.concat(
            [self._source_data, pd.DataFrame(rows, columns=self._source_data.columns)]
        ).reset_index(drop=True)
        (
            self._parsed_source_data,
            self._sliced_data,
            self._parsed_active_data,
        ) = self._prepare_working_data()
        self._rows_total = len(self._parsed_source_data["data"])
        DataJson()[self.widget_id]["data"] = list(self._parsed_active_data["data"])
        DataJson()[self.widget_id]["total"] = self._rows_total
        DataJson().send_changes()
        self._maybe_update_selected_row()

    def pop_row(self, index: Optional[int] = -1) -> List:
        """Removes a row from the table at the specified position and returns it.

        :param index: Index of the row to remove
        :type index: Optional[int]
        :return: Removed row
        :rtype: List
        """
        index = (
            len(self._parsed_source_data["data"]) - 1
            if index > len(self._parsed_source_data["data"]) or index < 0
            else index
        )

        if len(self._parsed_source_data["data"]) != 0:
            popped_row = self._source_data.loc[index].values
            self._source_data = self._source_data.drop(index)
            (
                self._parsed_source_data,
                self._sliced_data,
                self._parsed_active_data,
            ) = self._prepare_working_data()
            self._rows_total = len(self._parsed_source_data["data"])
            DataJson()[self.widget_id]["data"] = list(self._parsed_active_data["data"])
            DataJson()[self.widget_id]["total"] = self._rows_total
            self._maybe_update_selected_row()
            return popped_row

    def clear(self) -> None:
        """Clears the table data."""
        self._source_data = pd.DataFrame(columns=self._columns_first_idx)
        self._parsed_source_data = {"data": [], "columns": []}
        self._sliced_data = pd.DataFrame(columns=self._columns_first_idx)
        self._parsed_active_data = {"data": [], "columns": []}
        self._rows_total = 0
        DataJson()[self.widget_id]["data"] = {}
        DataJson()[self.widget_id]["total"] = 0
        DataJson().send_changes()
        self._maybe_update_selected_row()

    def row_click(self, func: Callable[[ClickedRow], Any]) -> Callable[[], None]:
        """Decorator for function that handles row click event.

        :param func: Function that handles row click event
        :type func: Callable[[ClickedRow], Any]
        :return: Decorated function
        :rtype: Callable[[], None]
        """
        row_clicked_route_path = self.get_route_path(FastTable.Routes.ROW_CLICKED)
        server = self._sly_app.get_server()

        self._row_click_handled = True
        self._is_row_clickable = True
        DataJson()[self.widget_id]["options"]["isRowClickable"] = self._is_row_clickable
        DataJson().send_changes()

        if self._cell_click_handled is True:
            message = "An exception occurred due to conflicting event listeners: 'row_click' and 'cell_click'. To avoid errors, specify only one event listener. The 'cell_click' listener includes all information about the row as well."
            raise EventLinstenerError(message)

        @server.post(row_clicked_route_path)
        def _click():
            try:
                clicked_row = self.get_selected_row()
                if clicked_row is None:
                    return
                func(clicked_row)
            except Exception as e:
                logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": str(e)})
                raise e

        return _click

    def cell_click(self, func: Callable[[ClickedCell], Any]) -> Callable[[], None]:
        """Decorator for function that handles cell click event.

        :param func: Function that handles cell click event
        :type func: Callable[[ClickedCell], Any]
        :return: Decorated function
        :rtype: Callable[[], None]
        """
        cell_clicked_route_path = self.get_route_path(FastTable.Routes.CELL_CLICKED)
        server = self._sly_app.get_server()

        self._cell_click_handled = True
        self._is_cell_clickable = True
        DataJson()[self.widget_id]["options"]["isCellClickable"] = self._is_cell_clickable
        DataJson().send_changes()

        if self._row_click_handled is True:
            message = "An exception occurred due to conflicting event listeners: 'row_click' and 'cell_click'. To avoid errors, specify only one event listener. The 'cell_click' listener includes all information about the row as well."
            raise EventLinstenerError(message)

        @server.post(cell_clicked_route_path)
        def _click():
            try:
                clicked_cell = self.get_selected_cell()
                if clicked_cell is None:
                    return
                func(clicked_cell)
            except Exception as e:
                logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": str(e)})
                raise e

        return _click

    def _default_filter_function(self, data: pd.DataFrame, filter_value: Any) -> pd.DataFrame:
        return data

    def _filter_table_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter source data using a self._filter_function as filter function.
        To apply a custom filter function, use the set_filter method.

        :return: Filtered data
        :rtype: pd.DataFrame
        """
        filtered_data = self._filter_function(data, self._filter_value)
        return filtered_data

    def _filter(self, filter_value: Any) -> pd.DataFrame:
        filtered_data = self._source_data.copy()
        if filter_value is None:
            return filtered_data
        if self._filter_value != filter_value:
            self._active_page = 1
            StateJson()[self.widget_id]["page"] = self._active_page
            StateJson().send_changes()
        self._filter_value = filter_value
        filtered_data = self._filter_table_data(filtered_data)
        return filtered_data

    def filter(self, filter_value) -> None:
        self._filter_value = filter_value
        self._refresh()

    def _default_search_function(self, data: pd.DataFrame, search_value: str) -> pd.DataFrame:
        # Use map() for pandas >= 2.1.0, fallback to applymap() for older versions
        if hasattr(pd.DataFrame, "map"):
            data = data[data.map(lambda x: search_value in str(x)).any(axis=1)]
        else:
            data = data[data.applymap(lambda x: search_value in str(x)).any(axis=1)]
        return data

    def _search(self, search_value: str) -> pd.DataFrame:
        """Search source data for search value.

        :param search_value: Search value
        :type search_value: str
        :return: Filtered data
        :rtype: pd.DataFrame
        """
        # Use filtered_data if available, otherwise use source_data directly
        if self._filtered_data is not None:
            filtered_data = self._filtered_data.copy()
        else:
            filtered_data = self._source_data.copy()

        if search_value == "":
            self._search_str = search_value
            return filtered_data
        if self._search_str != search_value:
            self._active_page = 1
            StateJson()[self.widget_id]["page"] = self._active_page
            StateJson().send_changes()
        filtered_data = self._search_function(filtered_data, search_value)
        self._search_str = search_value
        return filtered_data

    def search(self, search_value: str) -> None:
        StateJson()[self.widget_id]["search"] = search_value
        StateJson().send_changes()
        self._refresh()

    def _default_sort_function(
        self,
        data: pd.DataFrame,
        column_idx: Optional[int],
        order: Optional[Literal["asc", "desc"]],
    ) -> pd.DataFrame:
        if order == "asc":
            ascending = True
        else:
            ascending = False
        try:
            column = data.columns[column_idx]
            # Try to convert to numeric for proper sorting
            numeric_column = pd.to_numeric(data[column], errors="coerce")

            # Check if column contains numeric data (has at least one non-NaN numeric value)
            if numeric_column.notna().sum() > 0:
                # Create temporary column for sorting
                data_copy = data.copy()
                data_copy["_sort_key"] = numeric_column
                # Sort by numeric values with NaN at the end
                data_copy = data_copy.sort_values(
                    by="_sort_key", ascending=ascending, na_position="last"
                )
                # Remove temporary column and return original data in sorted order
                data = data.loc[data_copy.index]
            else:
                # Sort as strings with NaN values at the end
                data = data.sort_values(by=column, ascending=ascending, na_position="last")
        except IndexError as e:
            e.args = (
                f"Sorting by column idx = {column_idx} is not possible, your table has only {len(data.columns)} columns with idx from 0 to {len(data.columns) - 1}",
            )
            raise e
        return data

    def sort(
        self,
        column_idx: Optional[int] = None,
        order: Optional[Literal["asc", "desc"]] = None,
        reset: bool = False,
    ) -> None:
        """Sorts table data by column index and order.

        :param column_idx: Index of the column to sort by. If None, keeps current column (unless reset=True).
        :type column_idx: Optional[int]
        :param order: Sorting order. If None, keeps current order (unless reset=True).
        :type order: Optional[Literal["asc", "desc"]]
        :param reset: If True, clears sorting completely. Default is False.
        :type reset: bool

        :Usage example:

        .. code-block:: python
            # Sorting examples
            sort(column_idx=0, order="asc") # sort by column 0 ascending
            sort(column_idx=1) # sort by column 1, keep current order
            sort(order="desc") # keep current column, change order to descending
            sort(reset=True) # clear sorting completely
        """
        # If reset=True, clear sorting completely
        if reset:
            self._sort_column_idx = None
            self._sort_order = None
        else:
            # Preserve current values if new ones are not provided
            if column_idx is not None:
                self._sort_column_idx = column_idx
            # else: keep current self._sort_column_idx

            if order is not None:
                self._sort_order = order
            # else: keep current self._sort_order

        self._validate_sort_attrs()

        # Always update StateJson with current values (including None)
        StateJson()[self.widget_id]["sort"]["column"] = self._sort_column_idx
        StateJson()[self.widget_id]["sort"]["order"] = self._sort_order

        # Apply filter, search, sort pipeline
        self._filtered_data = self._filter(self._filter_value)
        self._searched_data = self._search(self._search_str)
        self._rows_total = len(self._searched_data)
        self._sorted_data = self._sort_table_data(self._searched_data)
        self._sliced_data = self._slice_table_data(self._sorted_data, actual_page=self._active_page)
        self._parsed_active_data = self._unpack_pandas_table_data(self._sliced_data)

        # Update DataJson with sorted and paginated data
        DataJson()[self.widget_id]["data"] = list(self._parsed_active_data["data"])
        DataJson()[self.widget_id]["total"] = self._rows_total
        self._maybe_update_selected_row()
        StateJson().send_changes()

    def _prepare_json_data(self, data: dict, key: str):
        if key in ("data", "columns"):
            default_value = []
        elif key == "options":
            default_value = {}
        else:
            default_value = None

        source_data = data.get(key, default_value)

        # Normalize options format: convert "columnIndex" to "column"
        if key == "options" and source_data is not None:
            sort = source_data.get("sort", None)
            if sort is not None:
                column_idx = sort.get("columnIndex", None)
                if column_idx is not None:
                    sort["column"] = column_idx
                    sort.pop("columnIndex")

        return source_data

    def _validate_sort(
        self, column_idx: int = None, order: Optional[Literal["asc", "desc"]] = None
    ):
        if column_idx is not None and type(column_idx) is not int:
            logger.warning(
                f"Incorrect value of 'column_id': {type(column_idx)} is not 'int'. Set to 'None'"
            )
            return False
        if column_idx is not None and column_idx < 0:
            logger.warning(f"Incorrect value of 'column_id': {column_idx} < 0. Set to 'None'")
            return False
        if order is not None and order not in ["asc", "desc"]:
            logger.warning(
                f"Incorrect value of 'order': {order}. Value can be one of 'asc' or 'desc'. Set to 'None'"
            )
            return False
        return True

    def _validate_sort_attrs(self):
        if not self._validate_sort(column_idx=self._sort_column_idx):
            self._sort_column_idx = None
        if not self._validate_sort(order=self._sort_order):
            self._sort_order = None

    def _validate_input_data(self, data):
        input_data_type = type(data)

        if input_data_type not in self._supported_types:
            raise TypeError(
                f"Cannot parse input data, please use one of supported datatypes: {self._supported_types}\n"
                """
                            1. Pandas Dataframe \n
                            2. Python list with structure [
                                                            ['row_1_column_1', 'row_1_column_2', ...],
                                                            ['row_2_column_1', 'row_2_column_2', ...],
                                                            ...
                                                          ]
                            """
            )

    def _create_multi_idx_columns(self):
        if self._columns_first_idx is not None:
            self._columns_second_idx = list(range(len(self._columns_first_idx)))
            _columns_tuples = [self._columns_first_idx, self._columns_second_idx]
            multi_idx_columns = pd.MultiIndex.from_tuples(
                list(zip(*_columns_tuples)), names=["first", "second"]
            )
        else:
            multi_idx_columns = None
        return multi_idx_columns

    def _prepare_input_data(self, data):
        if isinstance(data, pd.DataFrame):
            source_data = copy.deepcopy(data)

            if self._columns_first_idx is None and isinstance(source_data.columns, pd.RangeIndex):
                self._columns_first_idx = None
                self._multi_idx_columns = None
                return source_data
            elif self._columns_first_idx is None and isinstance(source_data.columns, pd.Index):
                self._columns_first_idx = source_data.columns.tolist()
            self._multi_idx_columns = self._create_multi_idx_columns()
            source_data.columns = self._multi_idx_columns
        elif isinstance(data, (list, type(None))):
            self._multi_idx_columns = self._create_multi_idx_columns()
            source_data = self._sort_table_data(
                pd.DataFrame(data=data, columns=self._multi_idx_columns)
            )
        return source_data

    def _prepare_working_data(self):
        parsed_source_data = self._unpack_pandas_table_data(input_data=self._source_data)
        sliced_data = self._slice_table_data(self._source_data, self._active_page)
        parsed_active_data = self._unpack_pandas_table_data(sliced_data)
        return parsed_source_data, sliced_data, parsed_active_data

    def _get_pandas_unpacked_data(self, data: pd.DataFrame) -> dict:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Cannot parse input data, please use Pandas Dataframe as input data")

        # Create a copy for frontend display to avoid modifying source data
        display_data = data.copy()
        # Replace NaN and None with empty string only for display
        display_data = display_data.replace({np.nan: "", None: ""})

        # Handle MultiIndex columns - extract only the first level
        if isinstance(display_data.columns, pd.MultiIndex):
            columns = display_data.columns.get_level_values("first").tolist()
        else:
            columns = display_data.columns.to_list()

        unpacked_data = {
            "columns": columns,
            "data": display_data.values.tolist(),
        }
        return unpacked_data

    def _unpack_pandas_table_data(self, input_data: pd.DataFrame) -> dict:
        """
        Convert data to dict from pd.DataFrame

        """
        if len(input_data.columns) != 0:
            new_data = []
            row_idx_list = input_data.index.tolist()
            try:
                columns = input_data.columns.get_level_values("first").tolist()
            except KeyError:
                columns = []
            data = copy.deepcopy(self._get_pandas_unpacked_data(input_data))
            for idx, row in zip(row_idx_list, data["data"]):
                data_entity = {"idx": idx, "items": row}
                new_data.append(data_entity)
            data["data"] = new_data
            data["columns"] = columns
        else:
            data = {
                "columns": [],
                "data": [],
            }
        return data

    def _slice_table_data(self, input_data: pd.DataFrame, actual_page: int = 1) -> pd.DataFrame:
        """
        Prepare data rows for the active page according to all filters that have been set

        """
        if input_data is not None:
            data: pd.DataFrame = copy.deepcopy(input_data)
            end_idx = self._page_size * actual_page
            if end_idx == self._page_size:
                start_idx = 0
            else:
                start_idx = end_idx - self._page_size
            data = data.iloc[start_idx:end_idx]
        return data

    def _sort_table_data(
        self,
        input_data: pd.DataFrame,
        column_index: Optional[int] = None,
        sort_order: Optional[Literal["asc", "desc"]] = None,
    ) -> pd.DataFrame:
        """
        Apply sorting to received data

        """
        if column_index is None:
            column_index = self._sort_column_idx
        if sort_order is None:
            sort_order = self._sort_order

        if sort_order is None or column_index is None:
            return input_data  # unsorted

        data = copy.deepcopy(input_data)
        if input_data is None:
            return data

        data = self._sort_function(data=input_data, column_idx=column_index, order=sort_order)
        return data

    def _unpack_project_meta(self, project_meta: Union[ProjectMeta, dict]) -> dict:
        """
        Apply sorting to received data

        """
        if project_meta is not None:
            if isinstance(project_meta, ProjectMeta):
                project_meta = project_meta.to_json()
            elif not isinstance(project_meta, dict):
                raise TypeError(
                    f"Project meta possible types: dict, ProjectMeta. Instead got {type(project_meta)}"
                )
        return project_meta

    def _validate_fix_columns_value(self, fixed_columns: int) -> int:
        if fixed_columns is None:
            return None
        if isinstance(fixed_columns, int):
            if fixed_columns <= 0:
                logger.warning(
                    f"The value for 'fixed_columns' should be > 0, instead '{fixed_columns}' is obtained. Set the value to 'None'"
                )
                fixed_columns = None
            elif fixed_columns > 1:
                logger.warning(
                    f"Value for 'fixed_columns' is currently only supported as '1', instead '{fixed_columns}' is obtained. Set the value to '1'"
                )
                fixed_columns = 1
        else:
            logger.warning(
                f"The value for 'fixed_columns' should be 'int', instead '{type(fixed_columns)}' is obtained. Set the value to 'None'"
            )
            fixed_columns = None
        return fixed_columns

    def _validate_table_sizes(self, row):
        if len(row) != len(self._source_data.columns):
            raise ValueError(
                "Sizes mismatch:\n"
                f"Lenght of row -> {len(row)} != {len(self._source_data.columns)} <- leght of columns"
            )

    def _validate_row_values_types(self, row):
        failed_column_idxs = []
        failed_column_idx = 0
        for column, value in zip(self._source_data.columns, row):
            if len(self._source_data[column].values) == 0:
                continue
            col_type = type(self._source_data[column].values[0])
            if col_type == str and not isinstance(value, str):
                failed_column_idxs.append(
                    {
                        "idx": failed_column_idx,
                        "type": col_type.__name__,
                        "row_value_type": type(value).__name__,
                    }
                )
            elif col_type != str and isinstance(value, str):
                failed_column_idxs.append(
                    {
                        "idx": failed_column_idx,
                        "type": col_type.__name__,
                        "row_value_type": type(value).__name__,
                    }
                )
            failed_column_idx += 1
        if len(failed_column_idxs) != 0:
            raise TypeError(
                f"Row contains values of types that do not match the data types in the columns: {failed_column_idxs}"
            )

    def update_cell_value(self, row: int, column: int, value: Any) -> None:
        """Updates the value of the cell in the table.

        :param row: Index of the row
        :type row: int
        :param column: Index of the column
        :type column: int
        :param value: New value
        :type value: Any
        """

        self._source_data.iat[row, column] = value
        self._parsed_source_data = self._unpack_pandas_table_data(self._source_data)
        self._sort_column_idx = StateJson()[self.widget_id]["sort"]["column"]
        self._sort_order = StateJson()[self.widget_id]["sort"]["order"]
        self._validate_sort_attrs()
        self._filtered_data = self._filter(self._filter_value)
        self._searched_data = self._search(self._search_str)
        self._rows_total = len(self._searched_data)
        self._sorted_data = self._sort_table_data(self._searched_data)

        increment = 0 if self._rows_total % self._page_size == 0 else 1
        max_page = self._rows_total // self._page_size + increment
        if (
            self._active_page > max_page
        ):  # active page is out of range (in case of the filtered data)
            self._active_page = max_page
            StateJson()[self.widget_id]["page"] = self._active_page

        self._sliced_data = self._slice_table_data(self._sorted_data, actual_page=self._active_page)
        self._parsed_active_data = self._unpack_pandas_table_data(self._sliced_data)
        DataJson()[self.widget_id]["data"] = list(self._parsed_active_data["data"])
        DataJson()[self.widget_id]["total"] = self._rows_total
        DataJson().send_changes()
        StateJson().send_changes()

    def selection_changed(self, func):
        """Decorator for function that handles selection change event.

        :param func: Function that handles selection change event
        :type func: Callable[[], Any]
        :return: Decorated function
        :rtype: Callable[[], None]
        """
        selection_changed_route_path = self.get_route_path(FastTable.Routes.SELECTION_CHANGED)
        server = self._sly_app.get_server()

        @server.post(selection_changed_route_path)
        def _selection_changed():
            if self._is_radio:
                selected_row = self.get_selected_row()
                func(selected_row)
            elif self._is_selectable:
                selected_rows = self.get_selected_rows()
                func(selected_rows)

        self._selection_changed_handled = True
        DataJson()[self.widget_id]["selectionChangedHandled"] = True
        DataJson().send_changes()
        return _selection_changed

    def select_row(self, idx: int):
        if not self._is_selectable and not self._is_radio:
            raise ValueError(
                "Table is not selectable. Set 'is_selectable' or 'is_radio' to True to use this method."
            )
        if idx < 0 or idx >= len(self._parsed_source_data["data"]):
            raise IndexError(
                f"Row index {idx} is out of range. Valid range is 0 to {len(self._parsed_source_data['data']) - 1}."
            )
        selected_row = self._parsed_source_data["data"][idx]
        self._selected_rows = [
            {"idx": idx, "row": selected_row.get("items", selected_row.get("row", None))}
        ]
        StateJson()[self.widget_id]["selectedRows"] = self._selected_rows
        page = idx // self._page_size + 1
        if self._active_page != page:
            self._active_page = page
            StateJson()[self.widget_id]["page"] = self._active_page
        self._refresh()

    def select_rows(self, idxs: List[int]):
        if not self._is_selectable:
            raise ValueError(
                "Table is not selectable. Set 'is_selectable' to True to use this method."
            )
        selected_rows = [
            self._parsed_source_data["data"][idx]
            for idx in idxs
            if 0 <= idx < len(self._parsed_source_data["data"])
        ]
        self._selected_rows = [
            {"idx": row["idx"], "row": row.get("items", row.get("row", None))}
            for row in selected_rows
        ]
        StateJson()[self.widget_id]["selectedRows"] = self._selected_rows
        StateJson().send_changes()

    def select_row_by_value(self, column, value: Any):
        """Selects a row by value in a specific column.
        The first column with the given name is used in case of duplicate column names.

        :param column: Column name to filter by
        :type column: str
        :param value: Value to select row by
        :type value: Any
        """
        if not self._is_selectable and not self._is_radio:
            raise ValueError(
                "Table is not selectable. Set 'is_selectable' to True to use this method."
            )
        if column not in self._columns_first_idx:
            raise ValueError(f"Column '{column}' does not exist in the table.")

        # Find the first column index with this name (in case of duplicates)
        column_idx = self._columns_first_idx.index(column)
        column_tuple = self._source_data.columns[column_idx]

        # Use column tuple to access the specific column
        idx = self._source_data[self._source_data[column_tuple] == value].index.tolist()
        if not idx:
            raise ValueError(f"No rows found with {column} = {value}.")
        if len(idx) > 1:
            raise ValueError(
                f"Multiple rows found with {column} = {value}. Please use select_rows_by_value method."
            )
        self.select_row(idx[0])

    def select_rows_by_value(self, column, values: List):
        """Selects rows by value in a specific column.
        The first column with the given name is used in case of duplicate column names.

        :param column: Column name to filter by
        :type column: str
        :param values: List of values to select rows by
        :type values: List
        """
        if not self._is_selectable:
            raise ValueError(
                "Table is not selectable. Set 'is_selectable' to True to use this method."
            )
        if column not in self._columns_first_idx:
            raise ValueError(f"Column '{column}' does not exist in the table.")

        # Find the first column index with this name (in case of duplicates)
        column_idx = self._columns_first_idx.index(column)
        column_tuple = self._source_data.columns[column_idx]

        # Use column tuple to access the specific column
        idxs = self._source_data[self._source_data[column_tuple].isin(values)].index.tolist()
        self.select_rows(idxs)

    def _read_custom_columns(self, columns: List[Union[str, tuple]]) -> None:
        if not columns:
            return
        self._columns = columns
        self._columns_options = self._columns_options or [{} for _ in columns]
        self._columns_data = []
        self._columns_first_idx = []
        for i, col in enumerate(columns):
            if isinstance(col, str):
                self._columns_first_idx.append(col)
                self._columns_data.append(self.ColumnData(name=col))
            elif isinstance(col, tuple):
                self._columns_first_idx.append(col[0])
                self._columns_data.append(
                    self.ColumnData(name=col[0], is_widget=True, widget=col[1])
                )
                self._columns_options[i]["customCell"] = True
            else:
                raise TypeError(f"Column name must be a string or a tuple, got {type(col)}")

        self._validate_sort_attrs()
