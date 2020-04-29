# coding: utf-8

import os
import json
import urllib.parse
from supervisely_lib.api.module_api import ApiField, ModuleApiBase

from supervisely_lib.collection.str_enum import StrEnum


class WidgetType(StrEnum):
    TABLE = 'table'
    PLOTLY = "plotly"
    MARKDOWN = "markdown"
    NOTIFICATION = "notification"


class NotificationType(StrEnum):
    INFO = 'info'
    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


class ReportApi(ModuleApiBase):
    def __init__(self, api):
        ModuleApiBase.__init__(self, api)

    def create(self, team_id, name, widgets, layout=""):
        response = self._api.post('reports.create', {ApiField.TEAM_ID: team_id,
                                                     ApiField.NAME: name,
                                                     ApiField.WIDGETS: widgets})
        return response.json()[ApiField.ID]

    def create_table(self, df, name, subtitle, per_page=20, pageSizes=[10, 20, 50, 100, 500], fix_columns=None):
        res = {
            "name": name,
            "subtitle": subtitle,
            "type": str(WidgetType.TABLE),
            "content": json.loads(df.to_json(orient='split')),
            "options": {
                "perPage": per_page,
                "pageSizes": pageSizes,
            }
        }
        if fix_columns is not None:
            res["options"]["fixColumns"] = fix_columns
        return res

    def create_notification(self, name, content, notification_type=NotificationType.INFO):
        return {
            "type": str(WidgetType.NOTIFICATION),
            "title": name,
            "content": content,
            "options": {
                "type": str(notification_type)
            }
        }

    def create_plotly(self, data_json, name, subtitle):
        data = data_json
        if type(data) is str:
            data = json.loads(data_json)
        elif type(data) is not dict:
            raise RuntimeError("type(data_json) is not dict")
        return {
            "name": name,
            "subtitle": subtitle,
            "type": str(WidgetType.PLOTLY),
            "content": data
        }

    def url(self, id):
        return urllib.parse.urljoin(self._api.server_address, 'reports/{}'.format(id))

