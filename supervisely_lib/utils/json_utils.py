# coding: utf-8

import json


def json_dump(obj, fpath, indent=False):
    indent = 4 if indent else None
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, indent=indent)


def json_load(path):
    return json.load(open(path, 'r', encoding='utf-8'))


def json_dumps(obj):
    return json.dumps(obj)


def json_loads(s):
    return json.loads(s, encoding='utf-8')
