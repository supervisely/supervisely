import copy
import json
import traceback
import numpy as np
import pandas as pd
from typing import Optional, Union

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
from supervisely.io.fs import get_file_ext


class PackerUnpacker:
    SUPPORTED_TYPES = tuple([dict, pd.DataFrame])

    @staticmethod
    def validate_sizes(unpacked_data):
        for row in unpacked_data["data"]:
            if len(row) != len(unpacked_data["columns"]):
                raise ValueError(
                    "Sizes mismatch:\n"
                    f'{len(row)} != {len(unpacked_data["columns"])}\n'
                    f"{row}\n"
                    f'{unpacked_data["columns"]}'
                )

    @staticmethod
    def unpack_data(data, unpacker_cb):
        unpacked_data = unpacker_cb(data)
        PackerUnpacker.validate_sizes(unpacked_data)
        return unpacked_data

    @staticmethod
    def pack_data(data, packer_cb):
        packed_data = packer_cb(data)
        return packed_data

    @staticmethod
    def dict_unpacker(data: dict):
        unpacked_data = copy.deepcopy(data)
        return unpacked_data

    @staticmethod
    def pandas_unpacker(data: pd.DataFrame):
        data = data.replace({np.nan: None})
        # data = data.astype(object).replace(np.nan, "-") # TODO: replace None later

        unpacked_data = {
            "columns": data.columns.to_list(),
            "data": data.values.tolist(),
        }
        return unpacked_data

    @staticmethod
    def dict_packer(data):
        packed_data = {"columns": data["columns"], "data": data["data"]}
        return packed_data

    @staticmethod
    def pandas_packer(data):
        data["data"] = [d["items"] for d in data["data"]]
        packed_data = pd.DataFrame(data=data["data"], columns=data["columns"])
        return packed_data


DATATYPE_TO_PACKER = {
    pd.DataFrame: PackerUnpacker.pandas_packer,
    dict: PackerUnpacker.dict_packer,
}

DATATYPE_TO_UNPACKER = {
    pd.DataFrame: PackerUnpacker.pandas_unpacker,
    dict: PackerUnpacker.dict_unpacker,
}


class DatasetNinjaTable(Widget):
    class Routes:
        ROW_CLICKED = "row_clicked_cb"
        CELL_CLICKED = "cell_clicked_cb"
        UPDATE_DATA = "update_data_cb"

    class ClickedDataRow:
        def __init__(
            self,
            row: list,
            row_index: int = None,
        ):
            self.row = row
            self.row_index = row_index

    class ClickedDataCell:
        def __init__(
            self,
            row: list,
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
        data: Optional[Union[dict, str]] = {},
        project_meta: Optional[Union[ProjectMeta, dict]] = None,
        clickable_rows: Optional[bool] = False,
        clickable_cells: Optional[bool] = False,
        width: Optional[str] = "auto",  # "200px", or "100%"
        widget_id: Optional[str] = None,
    ):
        """
        :param data: dataset or project data in different formats:
        1. Path to JSON file: 'statistics/class_balance.json'
        2. Python dict with structure {
                                        'columns': ['col_name_1', 'col_name_2', ...],
                                        'data': [
                                                    ['row_1_column_1', 'row_1_column_2', ...],
                                                    ['row_2_column_1', 'row_2_column_2', ...],
                                                    ...
                                                ],
                                        "columnsOptions": [{ "type": "class" }, { "maxValue": 10 }, ...],
                                        "options": {
                                                      "fixColumns": 1,
                                                      "sort": { "columnIndex": 1, "order": "desc" },
                                                      "pageSize": 10
                                                  }
                                      }
        """

        if clickable_rows is True and clickable_cells is True:
            raise AttributeError("You cannot use clickable rows and cells at the same time")

        self._supported_types = PackerUnpacker.SUPPORTED_TYPES
        self._row_click_handled = False
        self._cell_click_handled = False
        self._input_data = self._validate_input_data(data)
        self._columns_first_idx = self._prepare_input_data("columns")
        self._columns_options = self._prepare_input_data("columnsOptions")
        self._table_options = self._prepare_input_data("options")
        self._sorted_data = None
        self._filtered_data = None
        self._active_page = 1
        self._default_page_size = 10
        self._width = width
        self._selected_row = None
        self._selected_cell = None
        self._clickable_rows = clickable_rows
        self._clickable_cells = clickable_cells
        self._search_str = ""
        self._project_meta = self._unpack_project_meta(project_meta)

        # unpack table_options
        (
            self._page_size,
            self._fix_columns,
            self._sort_column_id,
            self._sort_order,
        ) = self._assign_table_options_attrs()

        # to avoid errors with the same names for columns
        self._multi_idx_columns = self._create_multi_idx_columns()

        # prepare source_data
        self._source_data = self._prepare_input_data("data")

        # prepare parsed_source_data, sliced_data, parsed_active_data
        (
            self._parsed_source_data,
            self._sliced_data,
            self._parsed_active_data,
        ) = self._assign_prepared_data_attrs()

        self._rows_total = len(self._parsed_source_data["data"])

        super().__init__(widget_id=widget_id, file_path=__file__)

        script_path = "./sly/css/app/widgets/dataset_ninja_table/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

        filter_changed_route_path = self.get_route_path(DatasetNinjaTable.Routes.UPDATE_DATA)
        server = self._sly_app.get_server()

        @server.post(filter_changed_route_path)
        def _filter_changed():
            try:
                StateJson().get_changes()
                self._active_page = StateJson()[self.widget_id]["page"]
                self._sort_order = StateJson()[self.widget_id]["sort"]["order"]
                self._sort_column_id = StateJson()[self.widget_id]["sort"]["column"]
                search_value = StateJson()[self.widget_id]["search"]
                self._filtered_data = self.search(search_value)
                self._rows_total = len(self._filtered_data)
                self._sorted_data = self._sort_table_data(self._filtered_data)
                self._sliced_data = self._slice_table_data(
                    self._sorted_data, actual_page=self._active_page
                )
                self._parsed_active_data = self._update_table_data(self._sliced_data)
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
                "column": self._sort_column_id,
                "order": self._sort_order,
            },
        }

    @property
    def fixed_columns_num(self) -> int:
        return self._fix_columns

    @fixed_columns_num.setter
    def fixed_columns_num(self, value: int):
        self._fix_columns = value
        DataJson()[self.widget_id]["options"]["fixColumns"] = self._fix_columns

    def read_json(self, data: dict, meta: dict = None) -> None:
        self._input_data = self._validate_input_data(data)
        self._columns_first_idx = self._prepare_input_data("columns")
        self._columns_options = self._prepare_input_data("columnsOptions")
        self._table_options = self._prepare_input_data("options")
        self._project_meta = self._unpack_project_meta(meta)
        self._multi_idx_columns = self._create_multi_idx_columns()
        self._source_data = self._prepare_input_data("data")
        self._sliced_data = self._slice_table_data(self._source_data)
        self._parsed_active_data = self._update_table_data(self._sliced_data)
        init_options = DataJson()[self.widget_id]["options"]
        init_options.update(self._table_options)
        DataJson()[self.widget_id]["data"] = self._parsed_active_data["data"]
        DataJson()[self.widget_id]["columns"] = self._parsed_active_data["columns"]
        DataJson()[self.widget_id]["columnsOptions"] = self._columns_options
        DataJson()[self.widget_id]["options"] = init_options
        DataJson()[self.widget_id]["total"] = len(self._source_data)
        DataJson()[self.widget_id]["pageSize"] = self._table_options.get(
            "pageSize", self._default_page_size
        )
        DataJson()[self.widget_id]["projectMeta"] = self._project_meta
        DataJson().send_changes()
        self.clear_selection()

    def to_json(self) -> dict:
        return self._get_packed_data(self._parsed_active_data, dict)

    def to_pandas(self) -> pd.DataFrame:
        return self._get_packed_data(self._parsed_active_data, pd.DataFrame)

    def clear_selection(self):
        StateJson()[self.widget_id]["selectedRow"] = None
        StateJson()[self.widget_id]["selectedCell"] = None
        StateJson().send_changes()

    def get_selected_row(self):
        row_data = StateJson()[self.widget_id]["selectedRow"]
        row_index = row_data["idx"]
        row = row_data["row"]

        if row_index is None or row is None:
            # click table header or clear selection
            return None

        return {
            "row": row,
            "row_index": row_index,
        }

    def get_selected_cell(self):
        cell_data = StateJson()[self.widget_id]["selectedCell"]
        row_index = cell_data["idx"]
        row = cell_data["row"]
        column_index = cell_data["column"]
        column_name = self._columns_first_idx[column_index]
        column_value = row[column_index]

        if column_index is None or row is None:
            # click table header or clear selection
            return None

        return {
            "row": row,
            "column_index": column_index,
            "row_index": row_index,
            "column_name": column_name,
            "column_value": column_value,
        }

    def insert_row(self, row, index=-1):
        PackerUnpacker.validate_sizes(
            {"columns": self._parsed_source_data["columns"], "data": [row]}
        )

        table_data = self._parsed_source_data["data"]
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
        ) = self._assign_prepared_data_attrs()
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
            ) = self._assign_prepared_data_attrs()
            self._rows_total = len(self._parsed_source_data["data"])
            DataJson()[self.widget_id]["data"] = self._parsed_active_data["data"]
            DataJson()[self.widget_id]["total"] = self._rows_total
            DataJson().send_changes()
            return popped_row

    def row_click(self, func):
        row_clicked_route_path = self.get_route_path(DatasetNinjaTable.Routes.ROW_CLICKED)
        server = self._sly_app.get_server()

        self._row_click_handled = True

        @server.post(row_clicked_route_path)
        def _click():
            try:
                value_dict = self.get_selected_row()
                if value_dict is None:
                    return
                datapoint = DatasetNinjaTable.ClickedDataRow(**value_dict)
                func(datapoint)
            except Exception as e:
                logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": str(e)})
                raise e

        return _click

    def cell_click(self, func):
        cell_clicked_route_path = self.get_route_path(DatasetNinjaTable.Routes.CELL_CLICKED)
        server = self._sly_app.get_server()

        self._cell_click_handled = True

        @server.post(cell_clicked_route_path)
        def _click():
            try:
                value_dict = self.get_selected_cell()
                if value_dict is None:
                    return
                datapoint = DatasetNinjaTable.ClickedDataCell(**value_dict)
                func(datapoint)
            except Exception as e:
                logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": str(e)})
                raise e

        return _click

    def search(self, search_value) -> pd.DataFrame:
        """
        Filter all rows in source data that contain filter_value in any column of row

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

    def sort(self, column_id: int = None, order: Optional[Literal["asc", "desc"]] = None):
        self._validate_sort(column_id, order)
        if column_id is not None:
            self._sort_column_id = column_id
            StateJson()[self.widget_id]["sort"]["column"] = column_id
        if order is not None:
            self._sort_order = order
            StateJson()[self.widget_id]["sort"]["order"] = order
        StateJson().send_changes()

    def _validate_sort(self, column_id: int = None, order: Optional[Literal["asc", "desc"]] = None):
        if column_id is not None and type(column_id) is not int:
            raise ValueError(f'Incorrect value of "column_id": {type(column_id)} is not "int".')
        if column_id is not None and column_id < 0:
            raise ValueError(f'Incorrect value of "column_id": {column_id} < 0')
        if order is not None and order not in ["asc", "desc"]:
            raise ValueError(
                f'Incorrect value of "order": {order}. Value can be one of "asc" or "desc".'
            )
        return True

    def _validate_input_data(self, data):
        if isinstance(data, str):
            if get_file_ext(data) == ".json":
                with open(data, "r") as json_file:
                    valid_data = json.load(json_file)
                return valid_data
        elif isinstance(data, dict):
            return data
        raise TypeError(f"Unsupported data type. Supported types: dict or .json file")

    def _prepare_input_data(self, key):
        if key in ("data", "columns"):
            default_value = {}
        else:
            default_value = None
        source_data = self._input_data.get(key, default_value)
        if key == "data":
            source_data = self._sort_table_data(
                pd.DataFrame(data=source_data, columns=self._multi_idx_columns)
            )
        return source_data

    def _assign_prepared_data_attrs(self):
        parsed_source_data = self._update_table_data(input_data=self._source_data)
        sliced_data = self._slice_table_data(self._source_data, self._active_page)
        parsed_active_data = self._update_table_data(sliced_data)
        return parsed_source_data, sliced_data, parsed_active_data

    def _create_multi_idx_columns(self):
        if self._columns_first_idx is not None:
            self._columns_second_idx = [
                "one" if i % 2 == 0 else "two" for i in range(len(self._columns_first_idx))
            ]
            _columns_tuples = [self._columns_first_idx, self._columns_second_idx]
            multi_idx_columns = pd.MultiIndex.from_tuples(
                list(zip(*_columns_tuples)), names=["first", "second"]
            )
        else:
            multi_idx_columns = None
        return multi_idx_columns

    def _assign_table_options_attrs(
        self,
    ):
        if self._table_options is not None:
            sort = self._table_options.get("sort", None)
            sort_column_id, sort_order = self._unpack_sort_attrs(sort)
            return (
                self._table_options.get("pageSize", self._default_page_size),
                self._table_options.get("fixColumns", None),
                sort_column_id,
                sort_order,
            )
        else:
            return self._default_page_size, None, None, None

    def _unpack_sort_attrs(self, sort):
        if sort is not None:
            sort_column_id = (
                sort["columnIndex"] if self._validate_sort(column_id=sort["columnIndex"]) else 0
            )
            sort_order = sort["order"] if self._validate_sort(order=sort["order"]) else "asc"
        else:
            sort_column_id = None
            sort_order = None
        return sort_column_id, sort_order

    def _update_table_data(self, input_data: pd.DataFrame) -> dict:
        """
        Convert data to dict from pd.DataFrame

        """
        if len(input_data.columns) != 0:
            new_data = []
            row_idx_list = input_data.index.tolist()
            columns = input_data.columns.get_level_values("first").tolist()
            data = copy.deepcopy(self._get_unpacked_data(input_data=input_data))
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
        if self._sort_order is None or self._sort_column_id is None:
            return input_data  # unsorted

        if input_data is not None:
            if self._sort_order == "asc":
                ascending = True
            else:
                ascending = False
            data: pd.DataFrame = copy.deepcopy(input_data)
            data = data.sort_values(by=data.columns[self._sort_column_id], ascending=ascending)
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

    def _get_packed_data(self, input_data, data_type):
        return PackerUnpacker.pack_data(data=input_data, packer_cb=DATATYPE_TO_PACKER[data_type])

    def _get_unpacked_data(self, input_data):
        input_data_type = type(input_data)

        if input_data_type not in self._supported_types:
            raise TypeError(
                f"Cannot parse input data, please use one of supported datatypes: {self._supported_types}\n"
                """
                            1. Pandas Dataframe \n
                            2. Python dict with structure {
                                        'columns': ['col_name_1', 'col_name_2', ...],
                                        'data': [
                                                            ['row_1_column_1', 'row_1_column_2', ...],
                                                            ['row_2_column_1', 'row_2_column_2', ...],
                                                            ...
                                                          ]
                                      }
                            """
            )

        return PackerUnpacker.unpack_data(
            data=input_data, unpacker_cb=DATATYPE_TO_UNPACKER[input_data_type]
        )
