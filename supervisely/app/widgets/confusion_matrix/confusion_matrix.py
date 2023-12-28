import copy
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.sly_logger import logger


class PackerUnpacker:
    SUPPORTED_TYPES = tuple([dict, pd.DataFrame])

    @staticmethod
    def validate_sizes(unpacked_data):
        if len(unpacked_data["data"]) != len(unpacked_data["classes"]):
            raise ValueError(
                "Sizes mismatch:\n"
                f'number of rows ({len(unpacked_data["data"])}) != number of columns ({len(unpacked_data["classes"])})'
            )

        for row in unpacked_data["data"]:
            if len(row) != len(unpacked_data["classes"]):
                raise ValueError(
                    "Sizes mismatch:\n"
                    f'{len(row)} != {len(unpacked_data["classes"])}\n'
                    f"{row}\n"
                    f'{unpacked_data["classes"]}'
                )

    @staticmethod
    def unpack_data(data, unpacker_cb, validate_sizes=True):
        unpacked_data = unpacker_cb(data)
        if validate_sizes:
            PackerUnpacker.validate_sizes(unpacked_data)
        return unpacked_data

    @staticmethod
    def pack_data(data, packer_cb):
        packed_data = packer_cb(data)
        return packed_data

    @staticmethod
    def dict_unpacker(data: dict):
        formatted_rows = []
        for origin_row in data["data"]:
            formatted_rows.append([{"value": element} for element in origin_row])

        unpacked_data = {"classes": data["columns"], "data": formatted_rows}

        return unpacked_data

    @staticmethod
    def pandas_unpacker(data: pd.DataFrame):
        data = data.where(pd.notnull(data), None)
        data = data.astype(object).replace(np.nan, "None")

        formatted_rows = []
        for origin_row in list(data.values):
            formatted_rows.append([{"value": element} for element in origin_row])

        unpacked_data = {"classes": data.columns.to_list(), "data": formatted_rows}
        return unpacked_data

    @staticmethod
    def dict_packer(data):
        unformatted_rows = []

        for origin_row in data["data"]:
            unformatted_rows.append([element["value"] for element in origin_row])

        packed_data = {
            "data": unformatted_rows,
            "columns": data["classes"],
        }
        return packed_data

    @staticmethod
    def pandas_packer(data):
        unformatted_rows = []

        for origin_row in data["data"]:
            unformatted_rows.append([element["value"] for element in origin_row])

        packed_data = pd.DataFrame(data=unformatted_rows, columns=data["classes"])
        return packed_data


DATATYPE_TO_PACKER = {pd.DataFrame: PackerUnpacker.pandas_packer, dict: PackerUnpacker.dict_packer}

DATATYPE_TO_UNPACKER = {
    pd.DataFrame: PackerUnpacker.pandas_unpacker,
    dict: PackerUnpacker.dict_unpacker,
}


class ConfusionMatrix(Widget):
    """ConfusionMatrix is a widget that display a given confusion matrix with color-coded visualization for better interpretation.
    It also shows row and column totals.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/charts-and-plots/confusionmatrix>`_
        (including screenshots and examples).


    :param data: Data of table in different formats (see usage example)
    :type data: Optional[Union[pd.DataFrame, Dict]]
    :param columns: List of column names
    :type columns: Optional[List[str]]
    :param x_label: Label for x axis
    :type x_label: Optional[str]
    :param y_label: Label for y axis
    :type y_label: Optional[str]
    :param widget_id: An unique identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import ConfusionMatrix

        # Option 1: Python dict
        confusion_matrix = ConfusionMatrix(
            data={
                "columns": ["class_1", "class_2", "class_3"],
                "data": [
                    ["1", "2", "3"],
                    ["4", "5", "6"],
                    ["7", "8", "9"],
                ],
            },
            x_label="Predicted Values",
            y_label="Actual Values",
        )

        # Option 2: Pandas DataFrame

        data = []
        for row in b:
            temp = [round(row * number, 1) for number in a]
            data.append(temp)

        a = [str(i) for i in a]
        b = [str(i) for i in b]

        data = pd.DataFrame(data=data, index=b, columns=a)
        confusion_matrix = ConfusionMatrix(data=data)

    """

    class Routes:
        CELL_CLICKED = "cell_clicked_cb"

    class ClickedDataPoint:
        """Represents data point of clicked cell in ConfusionMatrix.

        :param column_name: Name of column
        :type column_name: str
        :param column_index: Index of column
        :type column_index: int
        :param row_name: Name of row
        :type row_name: str
        :param row_index: Index of row
        :type row_index: int
        :param cell_value: Value of cell
        :type cell_value: Any
        """

        def __init__(
            self,
            column_name: str,
            column_index: int,
            row_name: str,
            row_index: int,
            cell_value: Any,
        ):
            self.column_index = column_index
            self.column_name = column_name
            self.row_index = row_index
            self.row_name = row_name
            self.cell_value = cell_value

    def __init__(
        self,
        data: Optional[Union[pd.DataFrame, Dict]] = None,
        columns: Optional[List[str]] = None,
        x_label: Optional[str] = "Predicted Values",
        y_label: Optional[str] = "Actual Values",
        widget_id: Optional[str] = None,
    ):
        self._supported_types = PackerUnpacker.SUPPORTED_TYPES

        self._parsed_data = None
        self._parsed_data_with_totals = {}
        self._data_type = None

        self._update_matrix_data(input_data=pd.DataFrame(data=data, columns=columns))

        self.x_label = x_label
        self.y_label = y_label

        self._loading = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict[str, Any]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - matrix_data: data of matrix
            - matrix_options: dictionary with options for matrix
                - selectable: if True, matrix will be selectable
                - horizontalLabel: label for horizontal axis
                - verticalLabel: label for vertical axis
            - loading: if True, loading animation will be shown

        :return: dictionary with widget data
        :rtype: Dict[str, Any]
        """

        return {
            "matrix_data": self._parsed_data_with_totals,
            "matrix_options": {
                "selectable": True,
                "horizontalLabel": self.x_label,
                "verticalLabel": self.y_label,
            },
            "loading": self._loading,
        }

    def get_json_state(self) -> Dict[str, Dict]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - selected_row: dictionary with empty dict

        :return: dictionary with widget state
        :rtype: Dict[str, Dict]
        """
        return {"selected_row": {}}

    def _update_matrix_data(self, input_data):
        if input_data is not None:
            self._parsed_data = copy.deepcopy(self.get_unpacked_data(input_data=input_data))
            self._data_type = type(input_data)
        else:
            self._parsed_data = {"classes": [], "data": []}
            self._data_type = dict
        self._calculate_totals()

    def _calculate_totals(self):
        matrix_data = []
        for origin_row in self._parsed_data["data"]:
            matrix_data.append([element["value"] for element in origin_row])

        totals_by_rows = np.asarray(["-" for _ in matrix_data]).reshape(-1, 1)
        totals_by_columns = np.asarray([["-" for _ in matrix_data]])

        try:
            matrix_data = np.matrix(matrix_data).astype(float)
            totals_by_rows = np.sum(matrix_data, axis=1).round(2)
            totals_by_columns = np.sum(matrix_data, axis=0).round(2)

            self._parsed_data_with_totals.update(self._calculate_max_values(matrix_data))

        except Exception as ex:
            logger.warning(f"Cannot calculate totals for matrix ({self.__class__.__name__}): {ex}")

        totals_by_columns = np.hstack([totals_by_columns, [[None]]])

        matrix_data = np.hstack([matrix_data, totals_by_rows])
        matrix_data = np.vstack([matrix_data, totals_by_columns])

        self._parsed_data_with_totals.update(
            self.get_unpacked_data(
                input_data={"columns": self._parsed_data["classes"], "data": matrix_data.tolist()},
                validate_sizes=False,
            )
        )

    def _get_packed_data(self, input_data, data_type):
        return PackerUnpacker.pack_data(data=input_data, packer_cb=DATATYPE_TO_PACKER[data_type])

    def get_unpacked_data(
        self, input_data, validate_sizes: Optional[bool] = True
    ) -> Dict[str, Any]:
        """Returns dictionary with unpacked data.

        :param input_data: Data of table in different formats (see usage example)
        :type input_data: Optional[Union[pd.DataFrame, Dict]]
        :param validate_sizes: If True, sizes of data will be validated, defaults to True
        :type validate_sizes: Optional[bool], optional
        :raises TypeError: If input data type is not supported
        :return: dictionary with unpacked data
        :rtype: Dict[str, Any]
        """
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
            data=input_data,
            unpacker_cb=DATATYPE_TO_UNPACKER[input_data_type],
            validate_sizes=validate_sizes,
        )

    def to_json(self) -> Dict[str, Any]:
        """Returns dictionary with widget data in JSON format.

        :return: dictionary with widget data in JSON format
        :rtype: Dict[str, Any]
        """
        return self._get_packed_data(self._parsed_data, dict)

    def to_pandas(self) -> pd.DataFrame:
        """Returns pandas DataFrame with widget data.

        :return: pandas DataFrame with widget data
        :rtype: pd.DataFrame
        """
        return self._get_packed_data(self._parsed_data, pd.DataFrame)

    def read_json(self, value: Dict[str, Any]) -> None:
        """Sets widget data from JSON.

        :param value: dictionary with widget data in JSON format
        :type value: Dict[str, Any]
        """
        self._update_matrix_data(input_data=value)
        DataJson()[self.widget_id]["table_data"] = self._parsed_data

    def read_pandas(self, value: pd.DataFrame) -> None:
        """Sets widget data from pandas DataFrame.

        :param value: pandas DataFrame with widget data
        :type value: pd.DataFrame
        """
        self._update_matrix_data(input_data=value)
        DataJson()[self.widget_id]["table_data"] = self._parsed_data

    def get_selected_cell(self, state: Dict[str, Dict]) -> Dict[str, Any]:
        """Returns dictionary with selected cell data.

        Dictionary contains the following fields:
            - row_index: index of row
            - col_index: index of column
            - row_data: data of row
            - cell_data: data of cell

        :param state: dictionary with widget state
        :type state: Dict[str, Dict]
        :return: dictionary with selected cell data
        :rtype: Dict[str, Any]
        """
        row_index = state[self.widget_id]["selected_row"].get("row")
        col_index = state[self.widget_id]["selected_row"].get("col")

        row_data = None
        cell_data = None

        if row_index is not None and col_index is not None:
            row_data = [element["value"] for element in self._parsed_data["data"][row_index]]
            cell_data = {
                "row_name": self._parsed_data["classes"][row_index],
                "col_name": self._parsed_data["classes"][col_index],
                "value": row_data[col_index],
            }

        return {
            "row_index": row_index,
            "col_index": col_index,
            "row_data": row_data,
            "cell_data": cell_data,
        }

    def _calculate_max_values(self, matrix_data):
        return {
            "diagonalMax": float(max(np.diagonal(matrix_data))),
            "maxValue": float(matrix_data.max()),
        }

    @property
    def loading(self) -> bool:
        """Returns True if loading animation is shown, False otherwise.

        :return: True if loading animation is shown, False otherwise
        :rtype: bool
        """
        return self._loading

    @loading.setter
    def loading(self, value: bool) -> None:
        """Sets loading animation.

        :param value: True if loading animation is shown, False otherwise
        :type value: bool
        """
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading

    def click(self, func: Callable[[ClickedDataPoint], Any]) -> Callable[[], None]:
        """Decorator for function that will be called when cell is clicked.

        :param func: function that will be called when cell is clicked
        :type func: Callable[[ClickedDataPoint], Any]
        :return: decorated function
        :rtype: Callable[[], None]
        """
        route_path = self.get_route_path(ConfusionMatrix.Routes.CELL_CLICKED)
        server = self._sly_app.get_server()

        @server.post(route_path)
        def _click():
            try:
                value_dict = self.get_selected_cell(StateJson())
                if value_dict is None:
                    return
                datapoint = ConfusionMatrix.ClickedDataPoint(
                    column_name=value_dict["cell_data"]["col_name"],
                    column_index=value_dict["col_index"],
                    row_name=value_dict["cell_data"]["row_name"],
                    row_index=value_dict["row_index"],
                    cell_value=value_dict["cell_data"]["value"],
                )
                func(datapoint)
            except Exception as e:
                logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": str(e)})
                raise e

        return _click
