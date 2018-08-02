# coding: utf-8

import os.path as osp

from jsonschema import Draft4Validator, validators

from ..json_utils import json_load


# may not work as expected, hm
def _extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(prop, subschema["default"])

        for error in validate_properties(validator, properties, instance, schema,):
            yield error

    return validators.extend(
        validator_class, {"properties": set_defaults},
    )


def _split_schema(src_schema):
    top_lvl = set(src_schema.keys()) - {'definitions'}

    def get_subschema(key):
        subs = src_schema[key].copy()
        subs.update({'definitions': src_schema['definitions']})
        return subs

    res = {k: get_subschema(k) for k in top_lvl}
    return res


class MultiTypeValidator(object):
    def __init__(self, schema_fpath):
        if not osp.isfile(schema_fpath):
            self.full_schema = self.subschemas = self.concrete_vtors = None
            return

        vtor_class = _extend_with_default(Draft4Validator)
        self.full_schema = json_load(schema_fpath)
        self.subschemas = _split_schema(self.full_schema)
        self.concrete_vtors = {k: vtor_class(v) for k, v in self.subschemas.items()}

    def val(self, type_name, obj):
        if self.concrete_vtors is None:
            raise RuntimeError('JSON validator is not defined. Type: {}'.format(type_name))
        try:
            self.concrete_vtors[type_name].validate(obj)
        except Exception as e:
            raise RuntimeError('Error occurred during JSON validation. Type: {}. Exc: {}'.format(
                type_name, str(e)
            )) from None  # suppress previous stacktrace, save all required info
