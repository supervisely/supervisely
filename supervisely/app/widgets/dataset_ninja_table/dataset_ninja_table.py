import copy
import io

import numpy as np
import pandas as pd
import re
import math
from typing import NamedTuple, Any, List, Optional, Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import DataJson
from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget
from supervisely.sly_logger import logger
import traceback
from fastapi.responses import StreamingResponse, RedirectResponse
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.project.project_meta import ProjectMeta


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
        # if "summaryRow" not in unpacked_data.keys():
        #     unpacked_data["summaryRow"] = None

        return unpacked_data

    @staticmethod
    def pandas_unpacker(data: pd.DataFrame):
        data = data.replace({np.nan: None})
        # data = data.astype(object).replace(np.nan, "-") # TODO: replace None later

        unpacked_data = {
            "columns": data.columns.to_list(),
            "data": data.values.tolist(),
            "summaryRow": None,
        }
        return unpacked_data

    @staticmethod
    def dict_packer(data):
        packed_data = {"columns": data["columns"], "data": data["data"]}
        if "summaryRow" in data.keys() and data["summaryRow"] is not None:
            packed_data["summaryRow"] = data["summaryRow"]
        return packed_data

    @staticmethod
    def pandas_packer(data):
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

"""
iris = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)
iris.insert(loc=0, column="index", value=np.arange(len(iris)))
table = sly.app.widgets.Table(data=iris, fixed_cols=1)

@table.click
def show_image(datapoint: sly.app.widgets.Table.ClickedDataRow):
    print("Column name = ", datapoint.column_name)
    print("Cell value = ", datapoint.cell_value)
    print("Row = ", datapoint.row)

"""


class DatasetNinjaTable(Widget):
    class Routes:
        ROW_CLICKED = "row_clicked_cb"
        UPDATE_SORT = "update_sort_cb"
        FILTER_CHANGED = "filter_changed_cb"
        UPDATE_DATA = "update_data"

    class ClickedDataRow:
        def __init__(
            self,
            row: dict,
            row_index: int = None,
        ):
            self.row = row
            self.row_index = row_index

    class ClickedDataCol:
        def __init__(
            self,
            colummn_index: int = None,
        ):
            self.Column_index = colummn_index

    def __init__(
        self,
        data: List[List],
        columns: Optional[List] = None,
        table_options: Optional[Dict] = None,
        columns_options: Optional[List[Dict]] = None,
        widget_id: Optional[str] = None,
        project_meta: ProjectMeta = None,
        clickable_rows: Optional[bool] = True,
        width: Optional[str] = "auto",  # "200px", or "100%"
    ):
        """
        :param data: Data of table in different formats:
        1. Pandas Dataframe or pd.DataFrame(
                                            data=data,
                                            columns=columns,
                                            table_options=table_options,
                                            columns_options=columns_options,
                                            )
        2. Python dict with structure {
                                        'columns': ['col_name_1', 'col_name_2', ...],
                                        'data': [
                                                    ['row_1_column_1', 'row_1_column_2', ...],
                                                    ['row_2_column_1', 'row_2_column_2', ...],
                                                    ...
                                                ]
                                      }
        """

        # fixed_cols: Optional[int] = None,
        # page_size: Optional[int] = 10,
        # sort_column_id: int = 0,
        # sort_order: Optional[Literal["asc", "desc"]] = "asc",

        self._supported_types = PackerUnpacker.SUPPORTED_TYPES

        # self._parsed_data = None
        self._data_type = None
        self._sorted_data = None
        self._filtered_data = None
        self._active_page = 1
        self._page_size = table_options["pageSize"]
        self._click_handled = False
        self._filter_handled = False
        self._sort_handled = False

        self._pd_data = pd.DataFrame(data=data, columns=columns)
        self._sliced_data = self._slice_table_data(self._pd_data, self._active_page)

        self._parsed_data = self._update_table_data(
            input_data=pd.DataFrame(data=data, columns=columns)
        )

        self._prepared_data = self._update_table_data(self._sliced_data)

        self._fix_columns = table_options["fixColumns"]
        self._width = width
        self._loading = False
        self._clickable_rows = clickable_rows
        self._columns_options = columns_options
        self._project_meta = project_meta

        self._rows_total = len(self._parsed_data["data"])

        # self._sort_column_id = (
        #     table_options["sort"]["columnIndex"]
        #     if self._validate_sort(column_id=table_options["sort"]["columnIndex"])
        #     else 0
        # )
        self._sort_column_id = 1
        self._sort_order = "desc"

        # self._sort_order = (
        #     table_options["sort"]["order"]
        #     if self._validate_sort(order=table_options["sort"]["order"])
        #     else "asc"
        # )

        self._search_str = ""

        super().__init__(widget_id=widget_id, file_path=__file__)

        script_path = "./sly/css/app/widgets/dataset_ninja_table/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

        server = self._sly_app.get_server()

        filter_changed_route_path = self.get_route_path(DatasetNinjaTable.Routes.UPDATE_DATA)

        @server.post(filter_changed_route_path)
        def _filter_changed():
            try:
                StateJson().get_changes()
                self._active_page = StateJson()[self.widget_id]["page"]
                self._sort_order = StateJson()[self.widget_id]["sort"]["order"]
                self._sort_column_id = StateJson()[self.widget_id]["sort"]["column"]
                search_value = StateJson()[self.widget_id]["search"]
                self._filtered_data = self.filter_rows(search_value)
                # self._filtered_data = self._update_table_data(filtered_data)
                self._rows_total = len(self._filtered_data)
                self._sorted_data = self._sort_table_data(self._filtered_data)
                self._sliced_data = self._slice_table_data(
                    self._sorted_data, actual_page=self._active_page
                )
                self._prepared_data = self._update_table_data(self._sliced_data)
                DataJson()[self.widget_id]["data"] = self._prepared_data["data"]
                DataJson()[self.widget_id]["total"] = self._rows_total
                DataJson().send_changes()
            except Exception as e:
                logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": str(e)})
                raise e

    def get_json_data(self):
        return {
            "data": self._prepared_data["data"],
            "columns": self._parsed_data["columns"],
            "projectMeta": self._project_meta,
            "columnsOptions": self._columns_options,
            "total": self._rows_total,
            "options": {
                "isRowClickable": self._clickable_rows,
                "fixColumns": self._fix_columns,
            },
            "pageSize": self._page_size,
        }

    def get_json_state(self):
        return {
            "search": self._search_str,
            "selectedRow": None,
            "page": self._active_page,
            "sort": {
                "column": self._sort_column_id,
                "order": self._sort_order,
            },
        }

    def _update_table_data(self, input_data):
        if input_data is not None:
            new_data = []
            idx = 1
            data = copy.deepcopy(self._get_unpacked_data(input_data=input_data))
            for d in data["data"]:
                data_entity = {"idx": idx, "items": d}
                new_data.append(data_entity)
                idx += 1
            data["data"] = new_data
        else:
            data = {"columns": [], "data": [], "summaryRow": None}
            self._data_type = dict
        return data

    def _slice_table_data(
        self,
        input_data,
        actual_page=1,
    ):
        if input_data is not None:
            data: pd.DataFrame = copy.deepcopy(input_data)
            end_idx = self._page_size * actual_page
            if end_idx == self._page_size:
                start_idx = 0
            else:
                start_idx = end_idx - self._page_size + 1
            data = data.iloc[start_idx:end_idx]
        return data

    def _sort_table_data(self, input_data):
        if input_data is not None:
            if self._sort_order == "asc":
                ascending = True
            else:
                ascending = False
            data: pd.DataFrame = copy.deepcopy(input_data)
            data = data.sort_values(by=data.columns[self._sort_column_id], ascending=ascending)
        return data

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

    @property
    def fixed_columns_num(self) -> int:
        return self._fix_columns

    @fixed_columns_num.setter
    def fixed_columns_num(self, value: int):
        self._fix_columns = value
        DataJson()[self.widget_id]["table_options"]["fixColumns"] = self._fix_columns

    @property
    def summary_row(self) -> List[Any]:
        # if "summaryRow" not in self._parsed_data.keys():
        #     return None
        return self._parsed_data["summaryRow"]

    @summary_row.setter
    def summary_row(self, value: List[Any]):
        cols_num = len(self._parsed_data["columns"])
        if len(value) < cols_num:
            value.extend([""] * (cols_num - len(value)))
        elif len(value) > cols_num:
            value = value[:cols_num]
        DataJson()[self.widget_id]["table_data"]["summaryRow"] = value

    def to_json(self) -> Dict:
        return self._get_packed_data(self._parsed_data, dict)

    def to_pandas(self) -> pd.DataFrame:
        return self._get_packed_data(self._parsed_data, pd.DataFrame)

    def read_json(self, value: dict) -> None:
        self._update_table_data(input_data=value)
        DataJson()[self.widget_id]["table_data"] = self._parsed_data
        DataJson().send_changes()
        self.clear_selection()

    def read_pandas(self, value: pd.DataFrame) -> None:
        self._update_table_data(input_data=value)
        DataJson()[self.widget_id]["table_data"] = self._parsed_data
        DataJson().send_changes()
        self.clear_selection()

    def insert_row(self, data, index=-1):
        PackerUnpacker.validate_sizes({"columns": self._parsed_data["columns"], "data": [data]})

        table_data = self._parsed_data["data"]
        index = len(table_data) if index > len(table_data) or index < 0 else index

        self._parsed_data["data"].insert(index, data)
        DataJson()[self.widget_id]["table_data"] = self._parsed_data
        DataJson().send_changes()

    def pop_row(self, index=-1):
        index = (
            len(self._parsed_data["data"]) - 1
            if index > len(self._parsed_data["data"]) or index < 0
            else index
        )

        if len(self._parsed_data["data"]) != 0:
            popped_row = self._parsed_data["data"].pop(index)
            DataJson()[self.widget_id]["table_data"] = self._parsed_data
            DataJson().send_changes()
            return popped_row

    def get_selected_row(self):
        # logger.debug(
        #     "Selected row",
        #     extra={"selected_row": state[self.widget_id]["selected_row"]},
        # )
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

    def click(self, func):
        row_clicked_route_path = self.get_route_path(DatasetNinjaTable.Routes.ROW_CLICKED)
        # update_sorting_column_route_path = self.get_route_path(DatasetNinjaTable.Routes.UPDATE_SORT)
        server = self._sly_app.get_server()

        # @server.post(update_sorting_column_route_path)
        # def _update_sorting_column():
        #     StateJson().send_changes()
        #     return

        self._click_handled = True

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

    def filter_rows(self, filter_value):
        # filter all rows that contain filter_value in any column of row

        filtered_data = self._pd_data.copy()
        if filter_value == "":
            return filtered_data
        else:
            filtered_data = filtered_data[
                filtered_data.applymap(lambda x: filter_value in str(x)).any(axis=1)
            ]
        return filtered_data

    @property
    def loading(self):
        return self._loading

    # @loading.setter
    # def loading(self, value: bool):
    #     self._loading = value
    #     DataJson()[self.widget_id]["loading"] = self._loading
    #     DataJson().send_changes()

    def clear_selection(self):
        StateJson()[self.widget_id]["selectedRow"] = {}
        StateJson().send_changes()

    def delete_row(self, key_column_name, key_cell_value):
        col_index = self._parsed_data["columns"].index(key_column_name)
        row_indices = []
        for idx, row in enumerate(self._parsed_data["data"]):
            if row[col_index] == key_cell_value:
                row_indices.append(idx)
        if len(row_indices) == 0:
            raise ValueError('Column "{key_column_name}" does not have value "{key_cell_value}"')
        if len(row_indices) > 1:
            raise ValueError(
                'Column "{key_column_name}" has multiple cells with the value "{key_cell_value}". Value has to be unique'
            )
        self.pop_row(row_indices[0])

    def update_cell_value(self, key_column_name, key_cell_value, column_name, new_value):
        key_col_index = self._parsed_data["columns"].index(key_column_name)
        row_indices = []
        for idx, row in enumerate(self._parsed_data["data"]):
            if row[key_col_index] == key_cell_value:
                row_indices.append(idx)
        if len(row_indices) == 0:
            raise ValueError('Column "{key_column_name}" does not have value "{key_cell_value}"')
        if len(row_indices) > 1:
            raise ValueError(
                'Column "{key_column_name}" has multiple cells with the value "{key_cell_value}". Value has to be unique'
            )

        col_index = self._parsed_data["columns"].index(column_name)
        self._parsed_data["data"][row_indices[0]][col_index] = new_value
        DataJson()[self.widget_id]["table_data"] = self._parsed_data
        DataJson().send_changes()

    def update_matching_cells(self, key_column_name, key_cell_value, column_name, new_value):
        key_col_index = self._parsed_data["columns"].index(key_column_name)
        row_indices = []
        for idx, row in enumerate(self._parsed_data["data"]):
            if row[key_col_index] == key_cell_value:
                row_indices.append(idx)

        col_index = self._parsed_data["columns"].index(column_name)
        for row_idx in row_indices:
            self._parsed_data["data"][row_idx][col_index] = new_value
        DataJson()[self.widget_id]["table_data"] = self._parsed_data
        DataJson().send_changes()

    def sort(self, column_id: int = None, order: Optional[Literal["asc", "desc"]] = None):
        self._validate_sort(column_id, order)
        if column_id is not None:
            self._sort_column_id = column_id
            StateJson()[self.widget_id]["sort"]["columnIdx"] = column_id
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
