# coding: utf-8

import string
import random
import re

def rand_str(length):
    chars = string.ascii_letters + string.digits  # [A-z][0-9]
    return ''.join((random.choice(chars)) for _ in range(length))

#@TODO: use in API? or remove
def generate_free_name(used_names, possible_name):
    res_name = possible_name
    new_suffix = 1
    while res_name in set(used_names):
        res_name = '{}_{:02d}'.format(possible_name, new_suffix)
        new_suffix += 1
    return res_name


def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def take_with_default(v, default):
    return v if v is not None else default
