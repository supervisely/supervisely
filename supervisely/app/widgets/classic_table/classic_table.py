import pandas as pd

import copy
import uuid
from typing import Union

import supervisely
from supervisely import Annotation
from supervisely.app import DataJson
from supervisely.app.widgets import Widget

from supervisely.sly_logger import logger


class PackerUnpacker:
    SUPPORTED_TYPES = tuple([dict, pd.DataFrame])

    @staticmethod
    def validate_sizes(unpacked_data):
        for row in unpacked_data['data']:
            if len(row) != len(unpacked_data['columns']):
                raise ValueError('Sizes mismatch:\n'
                                 f'{len(row)=} != {len(unpacked_data["columns"])=}\n'
                                 f'{row=}\n'
                                 f'{unpacked_data["columns"]=}')

    @staticmethod
    def unpack_data(data, unpacker_cb):
        unpacked_data = unpacker_cb(data)
        PackerUnpacker.validate_sizes(unpacked_data)
        return unpacked_data

    @staticmethod
    def dict_unpacker(data: dict):
        unpacked_data = {
            'columns': data['columns_names'],
            'data': data['values_by_rows']
        }

        return unpacked_data

    @staticmethod
    def pandas_unpacker(data: pd.DataFrame):
        unpacked_data = {
            'columns': data.columns.to_list(),
            'data': data.values.tolist()
        }
        return unpacked_data

    @staticmethod
    def dict_packer(data):
        raise NotImplementedError

    @staticmethod
    def pandas_packer(data):
        raise NotImplementedError


DATATYPE_TO_PACKER = {
    pd.DataFrame: PackerUnpacker.pandas_packer,
    dict: PackerUnpacker.dict_packer
}

DATATYPE_TO_UNPACKER = {
    pd.DataFrame: PackerUnpacker.pandas_unpacker,
    dict: PackerUnpacker.dict_unpacker
}


class ClassicTable(Widget):
    def __init__(self, data: PackerUnpacker.SUPPORTED_TYPES = None, widget_id: str = None):
        """
        :param data: Data of table in different formats:
        1. Pandas Dataframe
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

        self.update_table_data(input_data=data)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {**self._parsed_data}

    def get_json_state(self):
        return {'selected_row': 0}

    def update_table_data(self, input_data):
        if input_data is not None:
            self._parsed_data = self.get_unpacked_data(input_data=input_data)
            self._data_type = type(input_data)
        else:
            self._parsed_data = {
                'columns': [],
                'data': []
            }
            self._data_type = dict

    def get_unpacked_data(self, input_data):
        input_data_type = type(input_data)

        if input_data_type not in self._supported_types:
            raise TypeError(f'Cannot parse input data, please use one of supported datatypes: {self._supported_types}\n'
                            '''
                            1. Pandas Dataframe \n
                            2. Python dict with structure {
                                        'columns_names': ['col_name_1', 'col_name_2', ...],
                                        'values_by_rows': [
                                                            ['row_1_column_1', 'row_1_column_2', ...],
                                                            ['row_2_column_1', 'row_2_column_2', ...],
                                                            ...
                                                          ]
                                      }
                            ''')

        return PackerUnpacker.unpack_data(data=input_data,
                                          unpacker_cb=DATATYPE_TO_UNPACKER[input_data_type])

    @property
    def data(self):
        return self._parsed_data

    @data.setter
    def data(self, value):
        self.update_table_data(input_data=value)
        DataJson()[self.widget_id] = self.get_json_data()
