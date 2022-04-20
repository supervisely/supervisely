import pandas as pd

import copy
import uuid
from typing import Union

from varname import varname

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
    def pack_data(data, packer_cb):
        packed_data = packer_cb(data)
        return packed_data

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
        packed_data = {
            'columns_names': data['columns'],
            'values_by_rows': data['data']
        }
        return packed_data

    @staticmethod
    def pandas_packer(data):
        packed_data = pd.DataFrame(data=data['data'], columns=data['columns'])
        return packed_data


DATATYPE_TO_PACKER = {
    pd.DataFrame: PackerUnpacker.pandas_packer,
    dict: PackerUnpacker.dict_packer
}

DATATYPE_TO_UNPACKER = {
    pd.DataFrame: PackerUnpacker.pandas_unpacker,
    dict: PackerUnpacker.dict_unpacker
}


class ClassicTable(Widget):
    class Routes:
        def __init__(self,
                     app,
                     row_clicked_cb: object = None):
            self.app = app
            self.routes = {'row_clicked_cb': row_clicked_cb}

    def __init__(self, data: PackerUnpacker.SUPPORTED_TYPES = None, widget_routes: Routes = None,
                 widget_id: str = None):
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
        self.widget_id = varname(frame=1) if widget_id is None else widget_id

        self._supported_types = PackerUnpacker.SUPPORTED_TYPES

        self._parsed_data = None
        self._data_type = None

        self.update_table_data(input_data=data)

        self.available_routes = {}
        self.add_widget_routes(widget_routes)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {'table_data': self._parsed_data,
                'available_routes': self.available_routes}

    def get_json_state(self):
        return {'selected_row': None}

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

    def get_packed_data(self, input_data):
        return PackerUnpacker.pack_data(data=input_data,
                                        packer_cb=DATATYPE_TO_PACKER[self._data_type])

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
        return self.get_packed_data(self._parsed_data)

    @data.setter
    def data(self, value):
        self.update_table_data(input_data=value)
        DataJson()[self.widget_id]['table_data'] = self._parsed_data

    def add_widget_routes(self, routes: Routes):
        if routes is not None:
            for route_name, route_cb in routes.routes.items():
                if route_cb is not None:
                    routes.app.add_api_route(f'/{self.widget_id}/{route_name}', route_cb, methods=["POST"])
                    self.available_routes[route_name] = True
