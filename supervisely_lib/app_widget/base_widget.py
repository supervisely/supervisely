# coding: utf-8

from supervisely_lib.api.module_api import ApiField


class BaseWidget:
    widget_type = None

    @classmethod
    def _validate_type(cls, data):
        if cls.widget_type is None:
            raise ValueError("Child class has to define \"widget_type\" field")

        curr_type = data[ApiField.TYPE]
        if curr_type != str(cls.widget_type):
            raise ValueError("widget type is {!r}, but has to be {!r}".format(curr_type, str(cls.widget_type)))
