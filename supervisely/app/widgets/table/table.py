import copy

import fastapi
import numpy as np
import pandas as pd
from varname import varname
from typing import NamedTuple, Any, List

from supervisely.app import DataJson
from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget


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
        unpacked_data = {"columns": data["columns"], "data": data["data"]}

        return unpacked_data

    @staticmethod
    def pandas_unpacker(data: pd.DataFrame):
        data = data.where(pd.notnull(data), None)
        data = data.astype(object).replace(np.nan, "-")  # may be None

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
def show_image(datapoint: sly.app.widgets.Table.ClickedDataPoint):
    print("Column name = ", datapoint.column_name)
    print("Cell value = ", datapoint.cell_value)
    print("Row = ", datapoint.row)

"""


class Table(Widget):
    class Routes:
        CELL_CLICKED = "cell_clicked_cb"

    class ClickedDataPoint(NamedTuple):
        column_index: int
        column_name: str
        cell_value: Any
        row: dict

    def __init__(
        self,
        data=None,
        columns: list = None,
        fixed_cols: int = None,
        per_page: int = 15,
        page_sizes: List[int] = [10, 15, 30, 50, 100],
        width: str = "auto",  # "200px", or "100%"
        widget_id: str = None,
    ):
        """
        :param data: Data of table in different formats:
        1. Pandas Dataframe or pd.DataFrame(data=data, columns=columns)
        2. Python dict with structure {
                                        'columns_names': ['col_name_1', 'col_name_2', ...],
                                        'values_by_rows': [
                                                            ['row_1_column_1', 'row_1_column_2', ...],
                                                            ['row_2_column_1', 'row_2_column_2', ...],
                                                            ...
                                                          ]
                                      }
        """

        self._supported_types = PackerUnpacker.SUPPORTED_TYPES

        self._parsed_data = None
        self._data_type = None
        self._click_handled = False

        self._update_table_data(input_data=pd.DataFrame(data=data, columns=columns))

        self._per_page = per_page
        self._page_sizes = page_sizes
        self._fix_columns = fixed_cols
        self._width = width

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "table_data": self._parsed_data,
            "table_options": {
                "perPage": self._per_page,
                "pageSizes": self._page_sizes,
                "fixColumns": self._fix_columns,
            },
        }

    def get_json_state(self):
        return {"selected_row": {}}

    def _update_table_data(self, input_data):
        if input_data is not None:
            self._parsed_data = copy.deepcopy(
                self._get_unpacked_data(input_data=input_data)
            )
        else:
            self._parsed_data = {"columns": [], "data": []}
            self._data_type = dict

    def _get_packed_data(self, input_data, data_type):
        return PackerUnpacker.pack_data(
            data=input_data, packer_cb=DATATYPE_TO_PACKER[data_type]
        )

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
    def fixed_columns_num(self):
        return self._fix_columns

    @fixed_columns_num.setter
    def fixed_columns_num(self, value):
        self._fix_columns = value
        DataJson()[self.widget_id]["table_options"]["fixColumns"] = self._fix_columns

    def to_json(self) -> dict:
        return self._get_packed_data(self._parsed_data, dict)

    def to_pandas(self) -> pd.DataFrame:
        return self._get_packed_data(self._parsed_data, pd.DataFrame)

    def read_json(self, value: dict):
        self._update_table_data(input_data=value)
        DataJson()[self.widget_id]["table_data"] = self._parsed_data

    def read_pandas(self, value: pd.DataFrame):
        self._update_table_data(input_data=value)
        DataJson()[self.widget_id]["table_data"] = self._parsed_data

    def insert_row(self, data, index=-1):
        PackerUnpacker.validate_sizes(
            {"columns": self._parsed_data["columns"], "data": [data]}
        )

        table_data = self._parsed_data["data"]
        index = len(table_data) if index > len(table_data) or index < 0 else index

        self._parsed_data["data"].insert(index, data)
        DataJson()[self.widget_id]["table_data"] = self._parsed_data

    def pop_row(self, index=-1):
        index = (
            len(self._parsed_data["data"]) - 1
            if index > len(self._parsed_data["data"]) or index < 0
            else index
        )

        if len(self._parsed_data["data"]) != 0:
            popped_row = self._parsed_data["data"].pop(index)
            DataJson()[self.widget_id]["table_data"] = self._parsed_data
            return popped_row

    def get_selected_cell(self, state):
        row_index = state[self.widget_id]["selected_row"].get("selectedRow")
        column_name = state[self.widget_id]["selected_row"].get("selectedColumnName")
        column_index = state[self.widget_id]["selected_row"].get("selectedColumn")
        row = state[self.widget_id]["selected_row"].get("selectedRowData", {})

        return {
            "column_index": column_index,
            "column_name": column_name,
            "row": row,
            "cell_value": row[column_name],
        }

    def click(self, func):
        route_path = self.get_route_path(Table.Routes.CELL_CLICKED)
        server = self._sly_app.get_server()
        self._click_handled = True

        @server.post(route_path)
        def _click():
            value_dict = self.get_selected_cell(StateJson())
            datapoint = Table.ClickedDataPoint(**value_dict)
            func(datapoint)

        return _click
