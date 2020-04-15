# coding: utf-8
import json


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
    with open(filename, encoding='utf-8') as fin:
        return json.load(fin)


def dump_json_file(data, filename, indent=4):
    with open(filename, 'w') as fout:
        json.dump(data, fout, indent=indent)