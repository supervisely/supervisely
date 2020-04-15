# coding: utf-8

import os
import json
from supervisely_lib.api.module_api import ApiField, ModuleApiBase

from supervisely_lib.collection.str_enum import StrEnum


class WidgetType(StrEnum):
    TABLE = 'table'
    PLOTLY = "plotly"
    MARKDOWN = "markdown"


class ReportApi(ModuleApiBase):
    def __init__(self, api):
        ModuleApiBase.__init__(self, api)

    def create(self, team_id, widgets, layout=""):
        response = self._api.post('reports.create', {ApiField.TEAM_ID: team_id,
                                                     #ApiField.LAYOUT: layout,
                                                     ApiField.WIDGETS: widgets})
        return response.json()[ApiField.ID]

    def create_table(self, df, name, subtitle):
        return self._create_widget(name, subtitle, WidgetType.TABLE, json.loads(df.to_json(orient='split')))

    def _create_widget(self, name, subtitle, widget_type, content_json):
        return {
            "name": name,
            "subtitle": subtitle,
            "type": str(widget_type),
            "content": content_json
        }