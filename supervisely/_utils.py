# coding: utf-8

import os
import urllib
import string
import re
import base64
import hashlib
import json
import numpy as np
import random
from datetime import datetime
import copy
from typing import Optional
from supervisely.io import fs as sly_fs
from supervisely.sly_logger import logger

random.seed(datetime.now())


def rand_str(length):
    chars = string.ascii_letters + string.digits  # [A-z][0-9]
    return "".join((random.choice(chars)) for _ in range(length))


def generate_free_name(used_names, possible_name, with_ext=False, extend_used_names=False):
    res_name = possible_name
    new_suffix = 1
    while res_name in set(used_names):
        if with_ext is True:
            res_name = "{}_{:02d}{}".format(
                sly_fs.get_file_name(possible_name),
                new_suffix,
                sly_fs.get_file_ext(possible_name),
            )
        else:
            res_name = "{}_{:02d}".format(possible_name, new_suffix)
        new_suffix += 1
    if extend_used_names:
        used_names.add(res_name)
    return res_name


def generate_names(base_name, count):
    name = sly_fs.get_file_name(base_name)
    ext = sly_fs.get_file_ext(base_name)

    names = [base_name]
    for idx in range(1, count):
        names.append("{}_{:02d}{}".format(name, idx, ext))

    return names


def camel_to_snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def take_with_default(v, default):
    return v if v is not None else default


def batched(seq, batch_size=50):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def get_bytes_hash(bytes):
    return base64.b64encode(hashlib.sha256(bytes).digest()).decode("utf-8")


def get_string_hash(data):
    return base64.b64encode(hashlib.sha256(str.encode(data)).digest()).decode("utf-8")


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
        raise ValueError("Percent has to be in range [0; 100]")


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)


def _remove_sensitive_information(d: dict):
    new_dict = copy.deepcopy(d)
    fields = ["api_token", "API_TOKEN", "AGENT_TOKEN", "apiToken", "spawnApiToken"]
    for field in fields:
        if field in new_dict:
            new_dict[field] = "***"

    for parent_key in ["state", "context"]:
        if parent_key in new_dict and type(new_dict[parent_key]) is dict:
            for field in fields:
                if field in new_dict[parent_key]:
                    new_dict[parent_key][field] = "***"
    return new_dict


def validate_img_size(img_size):
    if not isinstance(img_size, (tuple, list)):
        raise TypeError(
            '{!r} has to be a tuple or a list. Given type "{}".'.format("img_size", type(img_size))
        )
    return tuple(img_size)


def is_development() -> bool:
    mode = os.environ.get("ENV", "development")
    if mode == "production":
        return False
    else:
        return True


def is_debug_with_sly_net() -> bool:
    mode = os.environ.get("DEBUG_WITH_SLY_NET")
    if mode is not None:
        return True
    else:
        return False


def is_production() -> bool:
    return not is_development()


def abs_url(relative_url: str) -> str:
    from supervisely.api.api import SERVER_ADDRESS

    server_address = os.environ.get(SERVER_ADDRESS, "")
    if server_address == "":
        logger.warn("SERVER_ADDRESS env variable is not defined")
    return urllib.parse.urljoin(server_address, relative_url)


def compress_image_url(
    url: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    quality: Optional[int] = 70,
) -> str:
    if width is None:
        width = ""
    if height is None:
        height = ""
    return url.replace(
        "/image-converter",
        f"/previews/{width}x{height},jpeg,q{quality}/image-converter",
    )


def get_preview_link(title="preview"):
    return (
        f'<a href="javascript:;">{title}<i class="zmdi zmdi-cast" style="margin-left: 5px"></i></a>'
    )


def get_datetime(value: str) -> datetime:
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")


def get_readable_datetime(value: str) -> str:
    dt = get_datetime(value)
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")
