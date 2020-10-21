# coding: utf-8
import json
import pandas as pd

class JsonSerializable:
    def to_json(self):
        """ Serialize to JSON-compatible dict.
        :return: dict
        """
        raise NotImplementedError()

    @classmethod
    def from_json(cls, data):
        """
        Deserialize from a JSON-compatible dict
        :param data: JSON-compatible dict
        :return: Parsed object
        """
        raise NotImplementedError()


def load_json_file(filename):
    '''
    Decoding data from json file with given filename
    :param filename: str
    :return: data in json format
    '''
    with open(filename, encoding='utf-8') as fin:
        return json.load(fin)


def dump_json_file(data, filename, indent=4):
    '''
    Write given data in json format in file with given name
    :param data: data to write in file in json format
    :param filename: str
    :param indent: int
    '''
    with open(filename, 'w') as fout:
        json.dump(data, fout, indent=indent)


def flatten_json(data, sep="."):
    df = pd.json_normalize(data, sep=sep)
    return df.to_dict(orient='records')[0]


def modify_keys(data, prefix=None, suffix=None):
    def _modify(k):
        res = k
        if prefix is not None:
            res = prefix + res
        if suffix is not None:
            res += suffix
        return res
    return {_modify(k): v for k, v in data.items()}
