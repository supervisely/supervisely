import fastapi
import pandas as pd
from varname import varname
import numpy as np
import copy

from supervisely.app import DataJson
from supervisely.app.widgets import Widget

from supervisely.sly_logger import logger


class PackerUnpacker:
    SUPPORTED_TYPES = tuple([dict, pd.DataFrame])

    @staticmethod
    def validate_sizes(unpacked_data):

        if len(unpacked_data['data']) != len(unpacked_data['classes']):
            raise ValueError('Sizes mismatch:\n'
                             f'number of rows ({len(unpacked_data["data"])}) != number of columns ({len(unpacked_data["classes"])})')

        for row in unpacked_data['data']:
            if len(row) != len(unpacked_data['classes']):
                raise ValueError('Sizes mismatch:\n'
                                 f'{len(row)} != {len(unpacked_data["classes"])}\n'
                                 f'{row}\n'
                                 f'{unpacked_data["classes"]}')

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
        for origin_row in data['data']:
            formatted_rows.append([{'value': element} for element in origin_row])

        unpacked_data = {
            'classes': data['columns'],
            'data': formatted_rows
        }

        return unpacked_data

    @staticmethod
    def pandas_unpacker(data: pd.DataFrame):
        data = data.where(pd.notnull(data), None)
        data = data.astype(object).replace(np.nan, 'None')

        formatted_rows = []
        for origin_row in list(data.values):
            formatted_rows.append([{'value': element} for element in origin_row])

        unpacked_data = {
            'classes': data.columns.to_list(),
            'data': formatted_rows
        }
        return unpacked_data

    @staticmethod
    def dict_packer(data):
        unformatted_rows = []

        for origin_row in data['data']:
            unformatted_rows.append([element['value'] for element in origin_row])

        packed_data = {
            'data': unformatted_rows,
            'columns': data['classes'],
        }
        return packed_data

    @staticmethod
    def pandas_packer(data):
        unformatted_rows = []

        for origin_row in data['data']:
            unformatted_rows.append([element['value'] for element in origin_row])

        packed_data = pd.DataFrame(data=unformatted_rows, columns=data['classes'])
        return packed_data


DATATYPE_TO_PACKER = {
    pd.DataFrame: PackerUnpacker.pandas_packer,
    dict: PackerUnpacker.dict_packer
}

DATATYPE_TO_UNPACKER = {
    pd.DataFrame: PackerUnpacker.pandas_unpacker,
    dict: PackerUnpacker.dict_unpacker
}


class ConfusionMatrix(Widget):
    class Routes:
        CELL_CLICKED = 'cell_clicked_cb'

    def __init__(self,
                 data: list = None,
                 columns: list = None,
                 x_label: str = 'Predicted Values',
                 y_label: str = 'Actual Values',
                 widget_id: str = None):
        """
        :param data: Data of table in different formats:
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
        self._supported_types = PackerUnpacker.SUPPORTED_TYPES

        self._parsed_data = None
        self._parsed_data_with_totals = {}
        self._data_type = None

        self._update_matrix_data(input_data=pd.DataFrame(data=data, columns=columns))

        self.x_label = x_label
        self.y_label = y_label

        self._loading = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            'matrix_data': self._parsed_data_with_totals,
            'matrix_options': {
                'selectable': len(DataJson().get(self.widget_id, {}).get('widget_routes', {})) > 0,
                'horizontalLabel': self.x_label,
                'verticalLabel': self.y_label
            },
            'loading': self._loading
        }

    def get_json_state(self):
        return {'selected_row': {}}

    def _update_matrix_data(self, input_data):
        if input_data is not None:
            self._parsed_data = copy.deepcopy(self.get_unpacked_data(input_data=input_data))
            self._data_type = type(input_data)
        else:
            self._parsed_data = {
                'classes': [],
                'data': []
            }
            self._data_type = dict
        self._calculate_totals()

    def _calculate_totals(self):
        matrix_data = []
        for origin_row in self._parsed_data['data']:
            matrix_data.append([element['value'] for element in origin_row])

        totals_by_rows = np.asarray(['-' for _ in matrix_data]).reshape(-1, 1)
        totals_by_columns = np.asarray([['-' for _ in matrix_data]])

        try:
            matrix_data = np.matrix(matrix_data).astype(float)
            totals_by_rows = np.sum(matrix_data, axis=1).round(2)
            totals_by_columns = np.sum(matrix_data, axis=0).round(2)

            self._parsed_data_with_totals.update(self._calculate_max_values(matrix_data))

        except Exception as ex:
            logger.warning(f'Cannot calculate totals for matrix ({self.__class__.__name__}): {ex}')

        totals_by_columns = np.hstack([totals_by_columns, [[None]]])

        matrix_data = np.hstack([matrix_data, totals_by_rows])
        matrix_data = np.vstack([matrix_data, totals_by_columns])

        self._parsed_data_with_totals.update(self.get_unpacked_data(input_data={
            'columns': self._parsed_data['classes'],
            'data': matrix_data.tolist()
        }, validate_sizes=False))

    def _get_packed_data(self, input_data, data_type):
        return PackerUnpacker.pack_data(data=input_data,
                                        packer_cb=DATATYPE_TO_PACKER[data_type])

    def get_unpacked_data(self, input_data, validate_sizes=True):
        input_data_type = type(input_data)

        if input_data_type not in self._supported_types:
            raise TypeError(f'Cannot parse input data, please use one of supported datatypes: {self._supported_types}\n'
                            '''
                            1. Pandas Dataframe \n
                            2. Python dict with structure {
                                        'columns': ['col_name_1', 'col_name_2', ...],
                                        'data': [
                                                            ['row_1_column_1', 'row_1_column_2', ...],
                                                            ['row_2_column_1', 'row_2_column_2', ...],
                                                            ...
                                                          ]
                                      }
                            ''')

        return PackerUnpacker.unpack_data(data=input_data,
                                          unpacker_cb=DATATYPE_TO_UNPACKER[input_data_type],
                                          validate_sizes=validate_sizes)

    def to_json(self) -> dict:
        return self._get_packed_data(self._parsed_data, dict)

    def to_pandas(self) -> pd.DataFrame:
        return self._get_packed_data(self._parsed_data, pd.DataFrame)

    def read_json(self, value: dict):
        self._update_matrix_data(input_data=value)
        DataJson()[self.widget_id]['table_data'] = self._parsed_data

    def read_pandas(self, value: pd.DataFrame):
        self._update_matrix_data(input_data=value)
        DataJson()[self.widget_id]['table_data'] = self._parsed_data

    def get_selected_cell(self, state):
        row_index = state[self.widget_id]['selected_row'].get('row')
        col_index = state[self.widget_id]['selected_row'].get('col')

        row_data = None
        cell_data = None

        if row_index is not None and col_index is not None:
            row_data = [element['value'] for element in self._parsed_data['data'][row_index]]
            cell_data = {
                'row_name': self._parsed_data['classes'][row_index],
                'col_name': self._parsed_data['classes'][col_index],
                'value': row_data[col_index]
            }

        return {
            'row_index': row_index,
            'col_index': col_index,
            'row_data': row_data,
            'cell_data': cell_data
        }

    def _calculate_max_values(self, matrix_data):
        return {
            "diagonalMax": float(max(np.diagonal(matrix_data))),
            "maxValue": float(matrix_data.max())
        }

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value: bool):
        self._loading = value
        DataJson()[self.widget_id]['loading'] = self._loading
