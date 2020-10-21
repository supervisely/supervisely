# coding: utf-8
import uuid

from supervisely_lib.api.module_api import ApiField
from supervisely_lib.app_widget.widget_type import WidgetType
from supervisely_lib.app_widget.base_widget import BaseWidget


class ButtonWidget(BaseWidget):
    widget_type = WidgetType.BUTTON

    @classmethod
    def create(cls, name, description, command, id=None):
        res = dict()
        res[ApiField.TYPE] = str(cls.widget_type)
        res[ApiField.NAME] = "button block title"
        res[ApiField.SUBTITLE] = "button block description"
        res[ApiField.CONTENT] = {
            ApiField.TITLE: name,
            ApiField.DESCRIPTION: description,
            ApiField.COMMAND: command,
        }
        res[ApiField.ID] = uuid.uuid4().hex if id is None else id
        return res