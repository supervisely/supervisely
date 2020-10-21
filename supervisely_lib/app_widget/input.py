# coding: utf-8
import uuid

# TODO: can not import from supervisely_lib.api.api import Api
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.app_widget.widget_type import WidgetType
from supervisely_lib.app_widget.base_widget import BaseWidget


class InputWidget(BaseWidget):
    widget_type = WidgetType.FORM

    @classmethod
    def get_value(cls, api, app_id, widget_id, allow_default=True, value_type=None):
        data = api.report.get_widget(report_id=app_id, widget_id=widget_id)
        cls._validate_type(data)
        value = data["content"].get("value", None)
        if value is None and allow_default is True:
            value = data["content"].get("defaultValue", None)
        if value_type is not None:
            value = value_type(value)
        return value

    @classmethod
    def create(cls, name, description, id=None, default_value=None):
        res = dict()
        res[ApiField.TYPE] = str(cls.widget_type)
        res[ApiField.NAME] = "input block title"
        res[ApiField.SUBTITLE] = "input block description"
        res[ApiField.CONTENT] = {
            ApiField.NAME: name,
            ApiField.DESCRIPTION: description
        }
        res[ApiField.ID] = uuid.uuid4().hex if id is None else id
        if default_value is not None:
            res[ApiField.CONTENT][ApiField.DEFAULT_VALUE] = default_value

        return res
