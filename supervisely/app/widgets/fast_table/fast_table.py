import copy
import traceback
import numpy as np
import pandas as pd
from typing import Optional, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import DataJson
from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget
from supervisely.sly_logger import logger
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.project.project_meta import ProjectMeta


class EventLinstenerError(Exception):
    def __init__(self, message="An exception occurred due to conflicting event listeners."):
        self.message = message
        super().__init__(self.message)


class FastTable(Widget):
    class Routes:
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
        width: Optional[str] = "auto",  # "200px", or "100%"
        widget_id: Optional[str] = None,
    ):
        """
        :param data: dataset or project data in different formats:
        1. Pandas Dataframe or pd.DataFrame(data=data, columns=columns)
        2. Python list with structure [
                                            ['row_1_column_1', 'row_1_column_2', ...],
                                            ['row_2_column_1', 'row_2_column_2', ...],
                                            ...
                                        ]
        """

        self._supported_types = tuple([pd.DataFrame, list, type(None)])
        self._row_click_handled = False
        self._cell_click_handled = False
        self._columns_first_idx = columns
        self._columns_options = columns_options
        self._sorted_data = None
        self._filtered_data = None
        self._active_page = 1
        self._width = width
        self._selected_row = None
        self._selected_cell = None
        self._clickable_rows = False
        self._clickable_cells = False
        self._search_str = ""
        self._project_meta = self._unpack_project_meta(project_meta)

        # table_options
        self._page_size = page_size
        self._fix_columns = self._validate_fix_columns_value(fixed_columns)
        self._sort_column_idx = sort_column_idx
        self._sort_order = sort_order
        self._validate_sort_attrs()

        # to avoid errors with the duplicated names in columns
        self._multi_idx_columns = None

        # prepare source_data
        self._validate_input_data(data)
        self._source_data = self._prepare_input_data(data)

        # prepare parsed_source_data, sliced_data, parsed_active_data
        (
            self._parsed_source_data,
            self._sliced_data,
            self._parsed_active_data,
        ) = self._prepare_working_data()

        self._rows_total = len(self._parsed_source_data["data"])

        super().__init__(widget_id=widget_id, file_path=__file__)

        script_path = "./sly/css/app/widgets/fast_table/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

        filter_changed_route_path = self.get_route_path(FastTable.Routes.UPDATE_DATA)
        server = self._sly_app.get_server()

        @server.post(filter_changed_route_path)
        def _filter_changed():
            try:
                self._active_page = StateJson()[self.widget_id]["page"]
                self._sort_order = StateJson()[self.widget_id]["sort"]["order"]
                self._sort_column_idx = StateJson()[self.widget_id]["sort"]["column"]
                search_value = StateJson()[self.widget_id]["search"]
                self._filtered_data = self.search(search_value)
                self._rows_total = len(self._filtered_data)
                self._sorted_data = self._sort_table_data(self._filtered_data)
                self._sliced_data = self._slice_table_data(
                    self._sorted_data, actual_page=self._active_page
                )
                self._parsed_active_data = self._unpack_pandas_table_data(self._sliced_data)
                DataJson()[self.widget_id]["data"] = self._parsed_active_data["data"]
                DataJson()[self.widget_id]["total"] = self._rows_total
                DataJson().send_changes()
            except Exception as e:
                logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": str(e)})
                raise e

    def get_json_data(self):
        return {
            "data": self._parsed_active_data["data"],
            "columns": self._parsed_source_data["columns"],
            "projectMeta": self._project_meta,
            "columnsOptions": self._columns_options,
            "total": self._rows_total,
            "options": {
                "isRowClickable": self._clickable_rows,
                "isCellClickable": self._clickable_cells,
                "fixColumns": self._fix_columns,
            },
            "pageSize": self._page_size,
        }

    def get_json_state(self):
        return {
            "search": self._search_str,
            "selectedRow": self._selected_row,
            "selectedCell": self._selected_cell,
            "page": self._active_page,
            "sort": {
                "column": self._sort_column_idx,
                "order": self._sort_order,
            },
        }

    @property
    def fixed_columns_num(self) -> int:
        return self._fix_columns

    @fixed_columns_num.setter
    def fixed_columns_num(self, value: int):
        self._fix_columns = self._validate_fix_columns_value(value)
        DataJson()[self.widget_id]["options"]["fixColumns"] = self._fix_columns

    @property
    def project_meta(self) -> dict:
        return self._project_meta

    @project_meta.setter
    def project_meta(self, meta: Union[ProjectMeta, dict]):
        self._project_meta = self._unpack_project_meta(meta)
        DataJson()[self.widget_id]["projectMeta"] = self._project_meta

    @property
    def page_size(self) -> int:
        return self._page_size

    @page_size.setter
    def page_size(self, size: int):
        self._page_size = size
        DataJson()[self.widget_id]["pageSize"] = self._page_size

    def read_json(self, data: dict, meta: dict = None) -> None:
        """
        Replace table data with options and project meta in the widget
        """
        self._columns_first_idx = self._prepare_json_data(data, "columns")
        self._columns_options = self._prepare_json_data(data, "columnsOptions")
        self._table_options = self._prepare_json_data(data, "options")
        self._project_meta = self._unpack_project_meta(meta)
        self._parsed_source_data = data.get("data", None)
        self._source_data = self._prepare_input_data(self._parsed_source_data)
        self._sliced_data = self._slice_table_data(self._source_data)
        self._parsed_active_data = self._unpack_pandas_table_data(self._sliced_data)
        init_options = DataJson()[self.widget_id]["options"]
        init_options.update(self._table_options)
        sort = init_options.pop("sort", {})
        page_size = init_options.pop("pageSize", 10)
        DataJson()[self.widget_id]["data"] = self._parsed_active_data["data"]
        DataJson()[self.widget_id]["columns"] = self._parsed_active_data["columns"]
        DataJson()[self.widget_id]["columnsOptions"] = self._columns_options
        DataJson()[self.widget_id]["options"] = init_options
        DataJson()[self.widget_id]["total"] = len(self._source_data)
        DataJson()[self.widget_id]["pageSize"] = page_size
        DataJson()[self.widget_id]["projectMeta"] = self._project_meta
        StateJson()[self.widget_id]["sort"] = sort
        DataJson().send_changes()
        StateJson().send_changes()
        self.clear_selection()

    def read_pandas(self, data: pd.DataFrame) -> None:
        """
        Replace table data (rows and columns) in the widget
        """
        self._source_data = self._prepare_input_data(data)
        self._sorted_data = self._sort_table_data(self._source_data)
        self._sliced_data = self._slice_table_data(self._sorted_data)
        self._parsed_active_data = self._unpack_pandas_table_data(self._sliced_data)
        DataJson()[self.widget_id]["data"] = self._parsed_active_data["data"]
        DataJson()[self.widget_id]["columns"] = self._parsed_active_data["columns"]
        DataJson()[self.widget_id]["total"] = len(self._source_data)
        DataJson().send_changes()
        self.clear_selection()

    def to_json(self, active_page=False) -> dict:
        """
        Export table data with current options as dict.

        :param active_page: Specifies the size of the data to be exported. If True - returns only the active page of the table
        :type active_page: bool
        :return: Table data with current options
        :rtype: dict
        """
        if active_page is True:
            temp_parsed_data = [d["items"] for d in self._parsed_active_data["data"]]
        else:
            temp_parsed_data = self._parsed_source_data
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
        """
        Export only table data (rows and columns) as Pandas Dataframe.

        :param active_page: Specifies the size of the data to be exported. If True - returns only the active page of the table
        :type active_page: bool
        :return: Table data
        :rtype: pd.DataFrame
        """
        if active_page is True:
            temp_parsed_data = [d["items"] for d in self._parsed_active_data["data"]]
        else:
            temp_parsed_data = self._parsed_source_data
        packed_data = pd.DataFrame(data=temp_parsed_data, columns=self._columns_first_idx)
        return packed_data

    def clear_selection(self):
        StateJson()[self.widget_id]["selectedRow"] = None
        StateJson()[self.widget_id]["selectedCell"] = None
        StateJson().send_changes()

    def get_selected_row(self):
        row_data = StateJson()[self.widget_id]["selectedRow"]
        row_index = row_data["idx"]
        row = row_data["row"]
        if row_index is None or row is None:
            return None
        return self.ClickedRow(row, row_index)

    def get_selected_cell(self):
        cell_data = StateJson()[self.widget_id]["selectedCell"]
        row_index = cell_data["idx"]
        row = cell_data["row"]
        column_index = cell_data["column"]
        column_name = self._columns_first_idx[column_index]
        column_value = row[column_index]
        if column_index is None or row is None:
            return None
        return self.ClickedCell(row, column_index, row_index, column_name, column_value)

    def insert_row(self, row, index=-1):
        self._validate_table_sizes(row)
        self._validate_row_values_types(row)
        table_data = self._parsed_source_data
        index = len(table_data) if index > len(table_data) or index < 0 else index

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
        DataJson()[self.widget_id]["data"] = self._parsed_active_data["data"]
        DataJson()[self.widget_id]["total"] = self._rows_total
        DataJson().send_changes()

    def pop_row(self, index=-1):
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
            DataJson()[self.widget_id]["data"] = self._parsed_active_data["data"]
            DataJson()[self.widget_id]["total"] = self._rows_total
            DataJson().send_changes()
            return popped_row

    def row_click(self, func):
        row_clicked_route_path = self.get_route_path(FastTable.Routes.ROW_CLICKED)
        server = self._sly_app.get_server()

        self._row_click_handled = True
        self._clickable_rows = True
        DataJson()[self.widget_id]["options"]["isRowClickable"] = self._clickable_rows
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

    def cell_click(self, func):
        cell_clicked_route_path = self.get_route_path(FastTable.Routes.CELL_CLICKED)
        server = self._sly_app.get_server()

        self._cell_click_handled = True
        self._clickable_cells = True
        DataJson()[self.widget_id]["options"]["isCellClickable"] = self._clickable_cells
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

    def search(self, search_value) -> pd.DataFrame:
        """
        Search source data for search_value

        """
        filtered_data = self._source_data.copy()
        if search_value == "":
            return filtered_data
        else:
            if self._search_str != search_value:
                self._active_page = 1
                StateJson()[self.widget_id]["page"] = self._active_page
                StateJson().send_changes()
            filtered_data = filtered_data[
                filtered_data.applymap(lambda x: search_value in str(x)).any(axis=1)
            ]
            self._search_str = search_value
        return filtered_data

    def sort(self, column_idx: int = None, order: Optional[Literal["asc", "desc"]] = None):
        self._sort_column_idx = column_idx
        self._sort_order = order
        self._validate_sort_attrs()
        if self._sort_column_idx is not None:
            StateJson()[self.widget_id]["sort"]["column"] = self._sort_column_idx
        if self._sort_order is not None:
            StateJson()[self.widget_id]["sort"]["order"] = self._sort_order
        self._filtered_data = self.search(self._search_str)
        self._rows_total = len(self._filtered_data)
        self._sorted_data = self._sort_table_data(self._filtered_data)
        self._sliced_data = self._slice_table_data(self._sorted_data, actual_page=self._active_page)
        self._parsed_active_data = self._unpack_pandas_table_data(self._sliced_data)
        DataJson()[self.widget_id]["data"] = self._parsed_active_data["data"]
        DataJson()[self.widget_id]["total"] = self._rows_total
        StateJson().send_changes()

    def _prepare_json_data(self, data: dict, key: str):
        if key in ("data", "columns"):
            default_value = []
        else:
            default_value = None
        source_data = data.get(key, default_value)
        if key == "data":
            source_data = self._sort_table_data(
                pd.DataFrame(data=source_data, columns=self._multi_idx_columns)
            )
        if key == "options":
            options = data.get(key, default_value)
            if options is not None:
                sort = options.get("sort", None)
                if sort is not None:
                    column_idx = sort.get("columnIndex", None)
                    if column_idx is not None:
                        sort["column"] = sort.get("columnIndex")
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
            raise TypeError(f"Cannot parse input data, please use Pandas Dataframe as input data")
        data = data.replace({np.nan: None})
        # data = data.astype(object).replace(np.nan, "-") # TODO: replace None later

        unpacked_data = {
            "columns": data.columns.to_list(),
            "data": data.values.tolist(),
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

    def _sort_table_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sorting to received data

        """
        if self._sort_order is None or self._sort_column_idx is None:
            return input_data  # unsorted

        if input_data is not None:
            if self._sort_order == "asc":
                ascending = True
            else:
                ascending = False
            data: pd.DataFrame = copy.deepcopy(input_data)
            try:
                data = data.sort_values(by=data.columns[self._sort_column_idx], ascending=ascending)
            except IndexError as e:
                e.args = (
                    f"Sorting by column idx = {self._sort_column_idx} is not possible, your table has only {len(data.columns)} columns with idx from 0 to {len(data.columns) - 1}",
                )
                raise e

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
            col_type = type(self._source_data[column][0])
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
