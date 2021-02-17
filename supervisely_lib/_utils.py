# coding: utf-8

import string
import re
import base64
import hashlib
import json
import numpy as np
import random
from datetime import datetime
import copy

from supervisely_lib.io import fs as sly_fs
from supervisely_lib.sly_logger import logger


random.seed(datetime.now())


def rand_str(length):
    chars = string.ascii_letters + string.digits  # [A-z][0-9]
    return ''.join((random.choice(chars)) for _ in range(length))


#@TODO: use in API? or remove
def generate_free_name(used_names, possible_name, with_ext=False):
    res_name = possible_name
    new_suffix = 1
    while res_name in set(used_names):
        if with_ext is True:
            res_name = '{}_{:02d}{}'.format(sly_fs.get_file_name(possible_name), new_suffix, sly_fs.get_file_ext(possible_name))
        else:
            res_name = '{}_{:02d}'.format(possible_name, new_suffix)
        new_suffix += 1
    return res_name


def generate_names(base_name, count):
    name = sly_fs.get_file_name(base_name)
    ext = sly_fs.get_file_ext(base_name)

    names = [base_name]
    for idx in range(1, count):
        names.append('{}_{:02d}{}'.format(name, idx, ext))

    return names


def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def take_with_default(v, default):
    return v if v is not None else default


def batched(seq, batch_size=50):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def get_bytes_hash(bytes):
    return base64.b64encode(hashlib.sha256(bytes).digest()).decode('utf-8')


def get_string_hash(data):
    return base64.b64encode(hashlib.sha256(str.encode(data)).digest()).decode('utf-8')


def unwrap_if_numpy(x):
    return x.item() if isinstance(x, np.number) else x


def _dprint(json_data):
    print(json.dumps(json_data))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


COMMUNITY = "community"
ENTERPRISE = "enterprise"


def validate_percent(value):
    if 0 <= value <= 100:
        pass
    else:
        raise ValueError('Percent has to be in range [0; 100]')


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def _remove_sensitive_information(d: dict):
    new_dict = copy.deepcopy(d)
    fields = ["api_token", "API_TOKEN", "AGENT_TOKEN", "apiToken"]
    for field in fields:
        if field in new_dict:
            new_dict[field] = "***"

    for parent_key in ["state", "context"]:
        if parent_key in new_dict and type(new_dict[parent_key]) is dict:
            for field in fields:
                if field in new_dict[parent_key]:
                    new_dict[parent_key][field] = "***"
    return new_dict